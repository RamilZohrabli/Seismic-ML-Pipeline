import copy
import time

import torch
import torch.nn.functional as F


def build_pixel_valid_mask(trace_valid_mask, height):
    """
    trace_valid_mask: (B, W)
    returns: (B, 1, H, W)
    """
    return trace_valid_mask[:, None, None, :].repeat(1, 1, height, 1)


def masked_bce_with_logits(logits, targets, pixel_valid_mask, pos_weight=20.0):
    """
    logits, targets, pixel_valid_mask: (B, 1, H, W)
    """
    pos_weight_tensor = torch.tensor(pos_weight, device=logits.device)

    loss_map = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight_tensor,
    )
    loss_map = loss_map * pixel_valid_mask
    denom = pixel_valid_mask.sum().clamp_min(1.0)
    return loss_map.sum() / denom


def masked_soft_dice_loss(logits, targets, pixel_valid_mask, eps=1e-6):
    probs = torch.sigmoid(logits)

    probs = probs * pixel_valid_mask
    targets = targets * pixel_valid_mask

    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def combined_loss(logits, targets, pixel_valid_mask, pos_weight=20.0, dice_weight=0.3):
    bce = masked_bce_with_logits(
        logits=logits,
        targets=targets,
        pixel_valid_mask=pixel_valid_mask,
        pos_weight=pos_weight,
    )
    dice = masked_soft_dice_loss(
        logits=logits,
        targets=targets,
        pixel_valid_mask=pixel_valid_mask,
    )
    total = bce + dice_weight * dice
    return total, bce.detach(), dice.detach()


def extract_pick_samples_from_logits(logits, threshold=0.5):
    """
    logits: (B, 1, H, W)
    return predicted pick index per trace: (B, W)
    """
    probs = torch.sigmoid(logits)
    B, C, H, W = probs.shape
    probs = probs[:, 0]  # (B, H, W)

    pred_idx = torch.zeros((B, W), dtype=torch.long, device=logits.device)

    for b in range(B):
        for w in range(W):
            col = probs[b, :, w]
            above = torch.where(col >= threshold)[0]
            if len(above) > 0:
                pred_idx[b, w] = above[0]
            else:
                pred_idx[b, w] = torch.argmax(col)

    return pred_idx


@torch.no_grad()
def compute_batch_mae_samples(logits, labels_sample, trace_valid_mask):
    """
    logits: (B,1,H,W)
    labels_sample: (B,W)
    trace_valid_mask: (B,W)
    """
    pred_idx = extract_pick_samples_from_logits(logits)

    valid = (labels_sample >= 0) & (trace_valid_mask > 0)

    if valid.sum() == 0:
        return None

    mae = torch.abs(pred_idx[valid] - labels_sample[valid]).float().mean()
    return mae.item()


def run_one_epoch(model, loader, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_mae = 0.0
    mae_count = 0

    for batch in loader:
        image = batch["image"].to(device).float()
        mask = batch["mask"].to(device).float()
        labels_sample = batch["labels_sample"].to(device).long()
        trace_valid_mask = batch["trace_valid_mask"].to(device).float()

        pixel_valid_mask = build_pixel_valid_mask(
            trace_valid_mask=trace_valid_mask,
            height=image.shape[2],
        )

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(image)

            loss, bce, dice = combined_loss(
                logits=logits,
                targets=mask,
                pixel_valid_mask=pixel_valid_mask,
                pos_weight=20.0,
                dice_weight=0.3,
            )

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_bce += bce.item()
        total_dice += dice.item()

        mae = compute_batch_mae_samples(
            logits=logits,
            labels_sample=labels_sample,
            trace_valid_mask=trace_valid_mask,
        )
        if mae is not None:
            total_mae += mae
            mae_count += 1

    n = len(loader)
    metrics = {
        "loss": total_loss / max(n, 1),
        "bce": total_bce / max(n, 1),
        "dice": total_dice / max(n, 1),
        "mae_samples": total_mae / max(mae_count, 1),
    }
    return metrics


def train_model(model, train_loader, val_loader, optimizer, device, epochs=5):
    history = []
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            train=False,
        )

        epoch_time = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae_samples": train_metrics["mae_samples"],
            "val_loss": val_metrics["loss"],
            "val_mae_samples": val_metrics["mae_samples"],
            "epoch_time_sec": epoch_time,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_mae={train_metrics['mae_samples']:.2f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_mae={val_metrics['mae_samples']:.2f} | "
            f"time={epoch_time:.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())

    return history, best_state