import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def _generate_data(skillinflow: pd.DataFrame, skilloutflow: pd.DataFrame, time_range: list, skills: list, class_num: int, min_length: int):
    '''generate training sequence data and demand/supply normalized company-position pairwise matrix data
    '''
    gap = skillinflow-skilloutflow
    inflow_matrices = _get_matrices(skillinflow, time_range)
    outflow_matrices = _get_matrices(skilloutflow, time_range)
    gap = _get_matrices(gap, time_range)

    # sampling
    data = []
    assert inflow_matrices.size() == outflow_matrices.size()
    max_length = inflow_matrices.size(0)
    min_length = min_length
    ''' 
    return x as [node_num, seq_len, start, end, len]
    return y as [node_num, label]
    '''
    # train
    for l in range(min_length, max_length, 1):  # l: x's length 2
        for i in range(0, max_length-l, 2): # 3
            x = (inflow_matrices[i: i+l, :].transpose(1,0), outflow_matrices[i: i+l, :].transpose(1,0))
            y = (inflow_matrices[i+l, :], outflow_matrices[i+l, :])
            g = gap[i: i+l, :].transpose(1,0)
            sample = {'X': x, 'Y': y, 'L': l, 'S': torch.arange(start=0,end=y[0].shape[0], dtype=torch.int64), 'G': g, 'Sta': i, 'End': i + l - 1}
            data.append(sample)
    # labeling
    in_ys = []
    out_ys = []
    for sample in data:
        d_y, s_y = sample['Y']
        in_ys.append(d_y)
        out_ys.append(s_y)
    in_val_range_list = _get_labels(torch.stack(in_ys), class_num)
    out_val_range_list = _get_labels(torch.stack(out_ys), class_num)
    for sample in data:
        d_y, s_y = sample['Y']
        d_y_label = _to_label(d_y, in_val_range_list, class_num)
        s_y_label = _to_label(s_y, out_val_range_list, class_num)
        sample['Y_Label'] = (d_y_label, s_y_label)
    seperate_by_y = {}
    for sample in data:
        if sample["End"] in seperate_by_y:
            seperate_by_y[sample["End"]].append(sample)
        else:
            seperate_by_y[sample["End"]] = [sample]
    data = []
    for key in seperate_by_y:
        data = data + seperate_by_y[key]
        
    return data, inflow_matrices, outflow_matrices


def _get_matrices(count, time_range, norm=True):
    '''get normalized company-position pairwise matrix data
    '''
    matrices = {}
    # build matrix
    for time in time_range:
        matrix = count[time].astype('float32')
        matrix[matrix.isna()] = 0.0
        if norm:
            matrix += 1e-6  # avoid divided by 0
        matrices[time] = torch.from_numpy(matrix.values)
    # stack data
    matrices = torch.stack(list(matrices.values()),dim=0).float()
    if norm:
        matrices = F.normalize(matrices, dim=0)
    return matrices

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