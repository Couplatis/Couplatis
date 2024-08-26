from loguru import logger
from tqdm import tqdm

from couplatis.config import BASE_DIR, Config
from couplatis.model import EpsNet
from couplatis.utils import get_test_loader, get_train_loader

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore


import torch
import os

import numpy as np


def run(config: Config):
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
    ckpt_folder = BASE_DIR.joinpath("data", "ckpt", suffix)

    log_path = BASE_DIR.joinpath("data", "eps_logs", "{}.log".format(suffix))
    logger.add(log_path, level="INFO")

    if ckpt_path is not None and os.path.exists(ckpt_path):
        logger.info("Loading checkpoints {}...".format(ckpt_path))
        net.load(ckpt_path)
        logger.success(f"Checkpoints {ckpt_path} loaded.")

    if not ckpt_folder.exists():
        ckpt_folder.mkdir(parents=True, exist_ok=True)

    best_epoch, best_acc = -1, -1
    for epoch_id in range(config.epoch):
        if (epoch_id + 1) in lr_milestone:
            net.alpha = net.alpha * lr_decay
            net.beta = net.beta * lr_decay
            logger.info("Learning rate decay to: {} and {}".format(net.alpha, net.beta))

        with tqdm(train_loader) as progress_bar:
            epoch_acc = 0.0
            for batch_id, batch in enumerate(train_loader):
                batch_acc = net.train(batch)
                epoch_acc += batch_acc
                progress_bar.update(1)
                progress_bar.set_description(
                    "Epoch: {}, Batch: {}/{}, Train Acc: {:.5f}".format(
                        epoch_id, batch_id, len(train_loader), batch_acc  # type: ignore
                    )
                )
                epoch_acc /= batch_id + 1
        ckpt_path = os.path.join(ckpt_folder, "epoch_{}.pth".format(str(epoch_id)))
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
            logger.info("Epoch: %d, Test Acc improved to: %.5f" % (epoch_id, acc))
        else:
            logger.info(
                "Epoch: %d, Test Acc is: %.5f, Best Test Acc is: %.5f in epoch: %d"
                % (epoch_id, acc, best_acc, best_epoch)
            )
