import logging

import torch


def run(use_cuda, model, criterion, optimizer, train_loader):
    device = torch.device("cuda:0" if use_cuda else "cpu")

    logging.debug("Start Training")

    model.train()
    avg_loss = 0.0
    iteration = 0

    for iteration, (user, item, label) in enumerate(train_loader):
        if use_cuda:
            user = user.cuda(device=device)
            item = item.cuda(device=device)
            label = label.float().cuda(device=device)

        model.zero_grad()
        prediction = model(user, item)
        loss = criterion(prediction, label.float())
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

    return avg_loss / iteration
