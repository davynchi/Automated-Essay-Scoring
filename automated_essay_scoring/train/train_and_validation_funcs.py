import time

import numpy as np
import torch

from ..common.constants import DEVICE
from ..common.dataset import collate
from ..common.utils import LOGGER, get_model_path, get_score
from .helper import AverageMeter, timeSince


def _move_to_device(data, device):
    """
    Recursively move tensor data to the specified device.

    Args:
        data (dict or torch.Tensor): Input data.
        device (torch.device): Device to move the data to.

    Returns:
        Data moved to the specified device.
    """
    if isinstance(data, dict):
        return {k: v.to(device) for k, v in data.items()}
    return data.to(device)


def valid_fn(valid_loader, valid_loader2, model, criterion, cfg):
    """
    Validate the model on two separate validation loaders.

    Args:
        valid_loader (DataLoader): Primary validation data loader.
        valid_loader2 (DataLoader): Secondary validation data loader.
        model (torch.nn.Module): The model to evaluate.
        criterion: Loss function.
        cfg: Configuration object.

    Returns:
        Tuple[float, np.array, np.array]:
            - Average loss over valid_loader,
            - Predictions from valid_loader,
            - Predictions from valid_loader2.
    """
    losses = AverageMeter()
    model.eval()
    all_preds = []
    start_time = time.time()

    # Process first validation loader with loss computation and logging.
    for step, (inputs, _, labels2) in enumerate(valid_loader):
        with torch.no_grad():
            inputs = collate(inputs)
            inputs = _move_to_device(inputs, DEVICE)
            labels2 = labels2.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels2)
            if cfg.base.gradient_accumulation_steps > 1:
                loss /= cfg.base.gradient_accumulation_steps
            batch_size = labels2.size(0)
            losses.update(loss.item(), batch_size)
            all_preds.append(outputs.sigmoid().to("cpu").numpy())

            if step % cfg.base.print_freq == 0 or step == (len(valid_loader) - 1):
                LOGGER.info(
                    f"EVAL: [{step + 1}/{len(valid_loader)}] "
                    f"Elapsed {timeSince(start_time, (step + 1) / len(valid_loader))} "
                    f"Loss: {losses.val:.4f}({losses.avg:.4f})"
                )
    predictions = np.concatenate(all_preds)

    # Process second validation loader for predictions only.
    all_preds_2 = []
    for _, (inputs, _, labels2) in enumerate(valid_loader2):
        with torch.no_grad():
            inputs = collate(inputs)
            inputs = _move_to_device(inputs, DEVICE)
            labels2 = labels2.to(DEVICE)
            outputs = model(inputs)
            all_preds_2.append(outputs.sigmoid().to("cpu").numpy())
    predictions2 = np.concatenate(all_preds_2)

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
    best_score,
    cfg,
):
    """
    Train the model for one epoch and validate at the end.

    Args:
        fold (int): Current fold number.
        train_loader (DataLoader): Training data loader.
        valid_loader (DataLoader): Primary validation data loader.
        valid_labels (np.array): Ground truth labels for primary validation.
        valid_loader2 (DataLoader): Secondary validation data loader.
        valid_labels2 (np.array): Ground truth labels for secondary validation.
        model (torch.nn.Module): The model to train.
        criterion: Loss function.
        optimizer: Optimizer.
        epoch (int): Current epoch number.
        scheduler: Learning rate scheduler.
        best_score (float): Best score achieved so far.
        cfg: Configuration object.

    Returns:
        float: Updated best score.
    """
    model.train()
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=cfg.base.apex)
    losses = AverageMeter()
    start_time = time.time()
    global_step = 0
    all_preds = []
    all_train_labels = []

    for step, (inputs, labels, labels2) in enumerate(train_loader):
        with torch.amp.autocast(DEVICE.type, enabled=cfg.base.apex):
            inputs = collate(inputs)
            inputs = _move_to_device(inputs, DEVICE)
            labels = labels.to(DEVICE)
            labels2 = labels2.to(DEVICE)
            batch_size = labels.size(0)
            outputs = model(inputs)
            loss = criterion(outputs, labels2)
            if cfg.base.gradient_accumulation_steps > 1:
                loss /= cfg.base.gradient_accumulation_steps

        losses.update(loss.item(), batch_size)
        all_preds.append(outputs.sigmoid().detach().to("cpu").numpy())
        all_train_labels.append(labels.detach().to("cpu").numpy())

        scaler.scale(loss).backward()

        if (step + 1) % cfg.base.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.base.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if cfg.base.batch_scheduler:
                scheduler.step()

        # Print training progress and perform validation at the end of the epoch.
        if step % (
            cfg.base.print_freq * cfg.base.gradient_accumulation_steps
        ) == 0 or step == (len(train_loader) - 1):
            current_lr = scheduler.get_lr()[0]
            LOGGER.info(
                f"Epoch: [{epoch + 1}][{step + 1} / {len(train_loader)}] "
                f"Elapsed {timeSince(start_time, (step + 1) / len(train_loader))} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"LR: {current_lr:.8f}"
            )

            # Trigger validation near the end of the epoch.
            if step > len(train_loader) - 2:
                predictions = np.concatenate(all_preds).astype(np.float32)
                train_labels_arr = np.concatenate(all_train_labels)
                train_score = get_score(train_labels_arr, predictions)
                avg_val_loss, val_predictions, val_predictions2 = valid_fn(
                    valid_loader, valid_loader2, model, criterion, cfg
                )
                score = get_score(valid_labels, val_predictions)
                score2 = get_score(valid_labels2, val_predictions2)
                elapsed = time.time() - start_time
                LOGGER.info(
                    f"Epoch_Step {epoch + 1}_{step} - avg_train_loss: {losses.avg:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
                )
                LOGGER.info(
                    f"Epoch {epoch + 1} - Train Score: {train_score:.4f} Val Score: {score:.4f} Val Score2: {score2:.4f}"
                )

                if best_score < score2:
                    best_score = score2
                    LOGGER.info(
                        f"Epoch_Step {epoch + 1}_{step} - Save Best Score: {best_score:.4f} Model"
                    )
                    model_path = get_model_path(cfg, fold)
                    torch.save(
                        {"model": model.state_dict(), "predictions": val_predictions},
                        model_path,
                    )

    return best_score
