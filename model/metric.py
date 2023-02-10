import torch
from torchmetrics.functional import classification, accuracy, auroc, f1_score, confusion_matrix


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
