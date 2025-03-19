import time

import numpy as np
import torch

from ..common.common import LOGGER
from ..common.dataset import collate
from ..common.model_utils import get_score
from .helper import AverageMeter, timeSince


def valid_fn(valid_loader, valid_loader2, model, criterion, device, cfg):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = time.time()
    for step, (inputs, _, labels2) in enumerate(valid_loader):
        print(f"val step: {step}")
        with torch.no_grad():
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels2 = labels2.to(device)
            y_preds = model(inputs)
            loss = criterion(y_preds, labels2)
        batch_size = labels2.size(0)
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to("cpu").numpy())
        if step % cfg.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step,
                    len(valid_loader),
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                )
            )
    predictions = np.concatenate(preds)

    preds = []
    for _, (inputs, _, labels2) in enumerate(valid_loader2):
        with torch.no_grad():
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels2 = labels2.to(device)
            y_preds = model(inputs)
        batch_size = labels2.size(0)
        preds.append(y_preds.sigmoid().to("cpu").numpy())
    predictions2 = np.concatenate(preds)

    return losses.avg, predictions, predictions2


def train_fn(
    fold,
    train_loader,
    valid_loader,
    valid_labels,
    valid_loader2,
    valid_labels2,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler,
    device,
    best_score,
    cfg,
):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
    losses = AverageMeter()
    start = time.time()
    global_step = 0
    preds = []
    train_labels = []
    for step, (inputs, labels, labels2) in enumerate(train_loader):
        print("step:", step)
        with torch.cuda.amp.autocast(enabled=cfg.apex):
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels = labels.to(device)
            labels2 = labels2.to(device)
            batch_size = labels.size(0)
            y_preds = model(inputs)
            loss = criterion(y_preds, labels2)
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().detach().to("cpu").numpy())
        train_labels.append(labels.detach().to("cpu").numpy())
        scaler.scale(loss).backward()
        # awp.attack_backward(inputs, labels, epoch)
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if cfg.batch_scheduler:
                scheduler.step()
        if step % (cfg.print_freq * cfg.gradient_accumulation_steps) == 0 or step == (
            len(train_loader) - 1
        ):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "LR: {lr:.8f}  ".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    lr=scheduler.get_lr()[0],
                )
            )

            if step > len(train_loader) - 2:  # как было изначально, поправь потом!
                # if True:
                predictions = np.concatenate(preds).astype(np.float32)
                train_labels = np.concatenate(train_labels)
                train_score = get_score(train_labels, predictions)
                avg_val_loss, predictions, predictions2 = valid_fn(
                    valid_loader, valid_loader2, model, criterion, device, cfg
                )
                score = get_score(valid_labels, predictions)
                score2 = get_score(valid_labels2, predictions2)
                elapsed = time.time() - start
                LOGGER.info(
                    f"Epoch_Step {epoch + 1}_{step} - avg_train_loss: {losses.avg:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
                )
                LOGGER.info(
                    f"Epoch {epoch + 1} - Train Score: {train_score:.4f} Val Score: {score:.4f} Val Score2: {score2:.4f}"
                )

                if best_score < score2:
                    best_score = score2
                    LOGGER.info(
                        f"Epoch_Step {epoch + 1}_{step} - Save Best Score: {best_score:.4f} Model\n"
                    )
                    torch.save(
                        {"model": model.state_dict(), "predictions": predictions},
                        cfg.path / f"{cfg.model.replace('/', '-')}_fold{fold}_best.pth",
                    )

    return best_score
