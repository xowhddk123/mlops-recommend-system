import torch


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return torch.tensor(1.0)
    return torch.tensor(0.0)


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = torch.nonzero(pred_items == gt_item).flatten()[0]
        return (index + 2).float().log2().reciprocal()
    return torch.tensor(0.0)
