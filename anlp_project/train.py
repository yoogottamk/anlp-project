from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from anlp_project.models.seq2seq import Seq2SeqRNN
from anlp_project.config import Config


def train_model(config: Config):
    input_size = None
    output_size = None

    model = Seq2SeqRNN(config, input_size, output_size)

    train_dataloader = None
    val_dataloader = None

    wandb_logger = WandbLogger()
    trainer = Trainer(logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=config.checkpoint_path)
    ...