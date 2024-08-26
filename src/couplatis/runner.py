"""
runner
"""

import os
import numpy as np
import torch

from loguru import logger
from tqdm import tqdm

from couplatis.config import BASE_DIR, Config
from couplatis.model import EpsNet
from couplatis.utils import get_test_loader, get_train_loader

try:
    from torch.utils.tensorboard.writer import SummaryWriter  # type: ignore
except Exception:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore


def run(config: Config):
    """run"""
    torch.manual_seed(42)
    train_loader = get_train_loader(config)
    test_loader = get_test_loader(config)

    writer = SummaryWriter(str(BASE_DIR.joinpath("data", "eps-sin2cos")))
    net = EpsNet(
        x_size=3072,
        y_size=4096,
        a_size=10,
        alpha=0.5,
        beta=0.5,
        js_mall=0.1,
        eps=0.1,
        config=config,
    )

    lr_milestone = [100, 200, 400]
    lr_decay = 0.5

    torch.manual_seed(24)
    suffix = "2023020101_epsNet"
    ckpt_path = None  # "ckpts/lhy/best_9956.pth"
    ckpt_folder = BASE_DIR.joinpath("data", "ckpts", suffix)

    log_path = BASE_DIR.joinpath("data", "eps_logs", f"{suffix}.log")
    logger.add(log_path, level="INFO")

    if ckpt_path is not None and os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoints {ckpt_path}...")
        net.load(ckpt_path)
        logger.success(f"Checkpoints {ckpt_path} loaded.")

    if not ckpt_folder.exists():
        ckpt_folder.mkdir(parents=True, exist_ok=True)

    best_epoch, best_acc = -1, -1
    for epoch_id in range(config.epoch):
        if epoch_id + 1 in lr_milestone:
            net.alpha = net.alpha * lr_decay
            net.beta = net.beta * lr_decay
            logger.info(f"Learning rate decay to: {net.alpha} and {net.beta}")

        with tqdm(train_loader) as progress_bar:
            epoch_acc = 0.0
            for batch_id, batch in enumerate(train_loader):
                batch_acc = net.train(batch)
                epoch_acc += batch_acc
                progress_bar.update(1)
                progress_bar.set_description(
                    f"Epoch: {epoch_id}, Batch: {batch_id}/{len(train_loader)}, Train Acc: {batch_acc:.5f}"
                )

                epoch_acc /= batch_id + 1
        ckpt_path = os.path.join(ckpt_folder, f"epoch_{epoch_id}.pth")
        writer.add_scalar("Train Acc", epoch_acc, epoch_id)
        net.save(ckpt_path)

        total, correct = 0.0, 0.0
        for batch in test_loader:
            a_pred = np.argmax(net.test(batch), axis=1)
            a_true = batch[1].numpy()
            correct += np.sum(a_pred == a_true)
            total += a_true.shape[0]

        acc = correct / total
        writer.add_scalar("Test acc", acc, epoch_id)
        if acc >= best_acc:
            best_acc = acc
            best_epoch = epoch_id
            logger.info(f"Epoch: {epoch_id}, Test Acc improved to: {acc:.5f}")
        else:
            logger.info(
                f"Epoch: {epoch_id}, Test Acc is: {acc:%.5f}, Best Test Acc is: {best_acc:.5f} in epoch:{best_epoch}"
            )
