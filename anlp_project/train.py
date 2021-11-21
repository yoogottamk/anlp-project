import logging
import multiprocessing
import os
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from anlp_project.config import Config
from anlp_project.datasets.europarl import EuroParl
from anlp_project.models.seq2seq import Seq2SeqRNN


def train_model(config: Config):
    dataset = EuroParl(config=config)

    # input is English, output is German
    input_size = dataset.de_vocab_size
    output_size = dataset.en_vocab_size

    logging.info(
        "Input size (German vocab size): %d; Output size (English vocab): %d",
        input_size,
        output_size,
    )

    model = Seq2SeqRNN(config, input_size, output_size)

    total_entries = len(dataset)
    train_ratio = 0.8
    train_length = int(total_entries * train_ratio)
    test_length = total_entries - train_length

    cpu_count = int(os.getenv("SLURM_CPUS_ON_NODE", str(multiprocessing.cpu_count())))
    train_data, test_data = random_split(
        dataset,
        [train_length, test_length],
        generator=torch.Generator().manual_seed(42),
    )
    train_dataloader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True, num_workers=cpu_count
    )
    # do not shuffle validation dataloader
    val_dataloader = DataLoader(
        test_data, batch_size=config.batch_size, num_workers=cpu_count
    )

    wandb_logger = WandbLogger()

    # -1 implies use all GPUs available
    gpu_count = -1 if torch.cuda.is_available() else 0
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=config.n_epochs,
        gpus=gpu_count,
        strategy="ddp",
        default_root_dir=str(Path(os.getcwd()) / "checkpoints")
    )
    trainer.fit(model, train_dataloader, val_dataloader)
