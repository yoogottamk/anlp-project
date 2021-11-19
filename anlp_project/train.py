import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader, random_split

from anlp_project.models.seq2seq import Seq2SeqRNN
from anlp_project.config import Config
from anlp_project.datasets.europarl import EuroParl


def train_model(config: Config):
    dataset = EuroParl(config=config)

    # input is English, output is German
    input_size = dataset.en_vocab_size
    output_size = dataset.de_vocab_size

    model = Seq2SeqRNN(config, input_size, output_size)

    europarl_data = EuroParl()

    total_entries = europarl_data.len
    train_ratio = 0.8
    train_length = int(total_entries * train_ratio)
    test_length = total_entries - train_length

    train_data, test_data = random_split(europarl_data, [train_length, test_length],
                                         generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=True)

    wandb_logger = WandbLogger()
    trainer = Trainer(logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=config.checkpoint_path)
