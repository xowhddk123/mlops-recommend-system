import logging

import torch
from torch import no_grad, topk, take, tensor, cat, mean

from metrics.metrics import ndcg, hit


def evaluate(use_cuda, model, dataloader, top_k):
    device = torch.device("cuda:0" if use_cuda else "cpu")

    logging.info("Calculating metrics...")

    model.eval()

    with no_grad():
        metric_result = {}
        logging.debug("Initializing Metrics...")
        hr_list = []
        ndcg_list = []
        valid_items = []
        total_recommends = []

        for idx, (user, item, _) in enumerate(dataloader):
            logging.debug("Load data to GPU...")
            if use_cuda:
                user = user.cuda(device=device)
                item = item.cuda(device=device)

            logging.debug("Run predict...")
            predictions = model(user, item)
            predictions = predictions.reshape(-1, 100)
            item = item.reshape(-1, 100)

            for p_idx, (itm, prediction) in enumerate(zip(item, predictions)):
                logging.debug(f"itm : {itm}")
                gt_item = itm[0]
                logging.debug(f"gt_item : {gt_item}")
                _, indices = topk(prediction, k=top_k, dim=-1)

                recommend = take(itm, indices)
                logging.debug(f"Recommend tensor : {recommend}")

                hr_list.append(tensor([hit(gt_item, recommend)]))
                ndcg_list.append(tensor([ndcg(gt_item, recommend)]))

                valid_items.append(itm.detach().cpu())
                total_recommends.append(recommend.detach().cpu())

        hr_val = mean(cat(hr_list, dim=0))
        ndcg_val = mean(cat(ndcg_list, dim=0))

        valid_items_val = cat(valid_items, dim=0)
        total_recommends_val = cat(total_recommends, dim=0)

        logging.debug(
            f"Unique Total Recommended Movie: {total_recommends_val.unique().size()}"
        )
        logging.debug(
            f"Unique Total Movie in Valid Dataset: {valid_items_val.unique().size()}"
        )

        var = float(
            total_recommends_val.unique().size()[0] / valid_items_val.unique().size()[0]
        )

        metric_result["HR"] = hr_val.item()
        metric_result["NDCG"] = ndcg_val.item()
        metric_result["VAR"] = var

        return metric_result
