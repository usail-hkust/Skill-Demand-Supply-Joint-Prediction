import numpy as np
import pandas as pd
import os
# from matplotlib import pyplot as plt

# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.api import ARIMA
import cupy as cp
from cuml.tsa.arima import ARIMA
# from statsmodels.tsa.api import VAR
# import statsmodels.api as sm
# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()

import torch
from torchmetrics.functional import classification, accuracy, auroc, f1_score, confusion_matrix
import torch.nn.functional as F
from tqdm import tqdm


import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())



data_dir = "/data/wchao/data/skill_inflow_outflow/"
dataset = "it"
subgraph_num=7446
subgraph_num = subgraph_num
startdate = pd.Timestamp(2017,10,1)
enddate = pd.Timestamp(2019,4,1)
period = 'M'
time_attr = pd.date_range(start=startdate, end=enddate, freq=period).to_period(period)
time_attr = time_attr.to_series().astype(str)
datasetinflow =  pd.read_csv(
    os.path.join(data_dir, f"skill_inflow_list_{dataset}.csv"),
    encoding='utf-8',
    header=0,
    on_bad_lines='warn',
)
datasetinflow.index = time_attr
datasetinflow.drop("time_attr", axis=1, inplace=True, errors="ignore")
datasetoutflow =  pd.read_csv(
    os.path.join(data_dir, f"skill_outflow_list_{dataset}.csv"),
    encoding='utf-8',
    header=0,
    on_bad_lines='warn'
)
datasetoutflow.index = time_attr
datasetoutflow.drop("time_attr", axis=1, inplace=True, errors="ignore")
datasetinflow = datasetinflow.iloc[:,:subgraph_num]
datasetoutflow = datasetoutflow.iloc[:,:subgraph_num]
time_range = datasetinflow.columns.value_counts().sort_index().index.tolist()



def _get_matrices(count, time_range):
    '''get normalized company-position pairwise matrix data
    '''
    matrices = {}
    # build matrix
    for time in time_range:
        matrix = count[time].astype('float32')
        matrix[matrix.isna()] = 0.0
        matrix += 1e-6  # avoid divided by 0
        matrices[time] = torch.from_numpy(matrix.values)
    # stack data
    matrices = torch.stack(list(matrices.values()),dim=0).float()
    # print(torch.std(matrices,dim=0,unbiased=True).shape)
    # print((torch.std(matrices,dim=0,unbiased=True)==0).any())
    # normalize data
    print(matrices.shape)
    matrices = F.normalize(matrices, dim=1)
    # print(matrices.shape)
    # print(matrices)
    # matrices_submean = torch.sub(matrices, torch.mean(matrices, dim=0))
    # matrices = torch.div(matrices_submean, torch.maximum(torch.std(matrices,dim=0,unbiased=True), torch.tensor(1)))
    return matrices

def accuracy_metric(output: torch.Tensor, labels: torch.Tensor, class_num=10):
    '''evaluate model in three classification metrics: accuracy, weighted f1 score and auroc
    '''
    pred = torch.argmax(output, dim=-1)
    # print(pred.shape, labels.shape)
    acc = classification.accuracy(pred, labels, task="multiclass", num_classes=class_num)
    # print(acc)
    return acc

def f1_metric(output: torch.Tensor, labels: torch.Tensor, class_num=10):
    pred = torch.argmax(output, dim=-1)
    weighted_f1 = f1_score(pred, labels, task="multiclass", average='weighted', num_classes=class_num)
    return weighted_f1

def aucroc_metric(output: torch.Tensor, labels: torch.Tensor, class_num=10):
    au = auroc(output, labels, task='multiclass', num_classes=class_num)
    return au

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


inflow_matrices = _get_matrices(datasetinflow, time_range)
outflow_matrices = _get_matrices(datasetoutflow, time_range)



def _get_labels(vec: torch.Tensor, class_num: int):
    '''split all range for data and get value range for each labels
    '''
    vec_tmp: torch.Tensor = vec.reshape(-1)
    vec_tmp, _ = vec_tmp.sort()
    n = len(vec_tmp)
    val_range_list: list = []
    for i in range(class_num):
        val_range_list.append(vec_tmp[(n // class_num) * i])
    val_range_list.append(vec_tmp[-1])
    return val_range_list

# def _get_label_SAX(vec: torch.Tensor, class_num: int):
    
def _to_label(vec: torch.Tensor, val_range_list: list, class_num: int):
    '''map continuous values to `class_num` discrete labels for `vec` using `val_range_list`
    '''

    def _to_label_(v: float, val_range_list: list, class_num: int):
        if v < val_range_list[0]:
            return 0
        for i in range(class_num):
            if val_range_list[i] <= v <= val_range_list[i + 1]:
                return i
        return class_num - 1

    return vec.clone().apply_(lambda x: _to_label_(x, val_range_list, class_num)).long()



data = []
val_range_list = []
class_num = 5
for i in range(1,3):
    supply_x = inflow_matrices[:,:-i]
    supply_y = inflow_matrices[:,inflow_matrices.shape[1]-i]
    demand_x = outflow_matrices[:,:-i]
    demand_y = outflow_matrices[:,outflow_matrices.shape[1]-i]
    sample = {'supply_x': supply_x, 'supply_y': supply_y, "demand_x": demand_x, "demand_y": demand_y}
    data.append(sample)

for sample in data:
    val_range_list_supply = _get_labels(sample["supply_y"], class_num)
    val_range_list_demand = _get_labels(sample["demand_y"], class_num)
    val_range_list.append({"supply": val_range_list_supply, "demand":val_range_list_demand})
    sample["supply_y_label"] = _to_label(sample["supply_y"], val_range_list_supply, class_num)
    sample["demand_y_label"] = _to_label(sample["demand_y"], val_range_list_demand, class_num)


sample = data[0]
pred = []
batch_size = 128
for i in range(3):
    print("batch:", i)
    x = np.nan_to_num(sample["supply_x"].numpy()[i*batch_size:(i+1)*batch_size,:].T, 0,posinf=1,neginf=1)
    model = ARIMA(cp.array(x), order=(3,1,2), fit_intercept=False)
    results = model.fit()
    pred.append(model.forecast(1))
temp = np.concatenate(pred)
print(temp)

print(model.forecast(1))

sup_or_dem = ["supply", "demand"]

sample = data[0]
sup_or_dem = ["supply", "demand"]
for idx, sample in enumerate(data):
    for sd in sup_or_dem:
        pred = []
        for i in range(int(sample["supply_x"].shape[0]/batch_size)+1):
            model = ARIMA(cp.array(sample["supply_x"].numpy()[i*batch_size:i+1*batch_size,:].T), order=(1,1,2), fit_intercept=False)
            results = model.fit()
            pred.append(model.forecast(1))
        temp = np.concatenate(pred)
        print(temp)
        output = _to_label(torch.tensor(pred), val_range_list[idx][sd], class_num)
        output = F.one_hot(output)
        sample[sd+"_y_output"] = output


sup_or_dem = ["supply", "demand"]
combined_output_list = []
combined_label_list = []
for sd in sup_or_dem:
    output_list = []
    label_list = []
    for idx, sample in enumerate(data):
        output_list.append(sample[sd+"_y_output"])
        label_list.append(sample[sd+"_y_label"])
    # print(output.shape)
    output = torch.cat(output_list,dim=1).squeeze().float()
    # print(output.shape)
    label = torch.cat(label_list,dim=0)
    # print(label.shape)
    print(sd+"_accuracy:", accuracy_metric(output,label,class_num).item())
    print(sd+"_f1:", f1_metric(output,label,class_num).item())
    print(sd+"auc", aucroc_metric(output,label,class_num).item())
    combined_output_list.append(output)
    combined_label_list.append(label)
combined_output = torch.cat(combined_output_list,dim=0)
combined_label = torch.cat(combined_label_list,dim=0)
print("combined_accuracy:", accuracy_metric(combined_output,combined_label,class_num).item())
print("combined_f1:", f1_metric(combined_output,combined_label,class_num).item())
print("combined_auc", aucroc_metric(combined_output,combined_label,class_num).item())