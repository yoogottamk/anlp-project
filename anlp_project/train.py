import logging
import multiprocessing
import os
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from fairseq.data.dictionary import Dictionary
from fairseq.models.transformer import TransformerModel
from fairseq.tasks.translation import TranslationTask

from anlp_project.config import Config
from anlp_project.datasets.europarl import EuroParl
from anlp_project.utils import get_checkpoint_dir


def get_transformer_model(args=None):
    if not args:
        args = {}
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.no_cross_attention = getattr(args, 'no_cross_attention', False)
    args.cross_self_attention = getattr(args, 'cross_self_attention', False)
    args.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    model = TransformerModel(args, encoder, decoder)
    return model



def train_model(config: Config):
    dataset = EuroParl(config=config)
    task = TranslationTask()
    model = get_transformer_model()

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
    checkpoint_path = get_checkpoint_dir(config)
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=config.n_epochs,
        gpus=gpu_count,
        strategy="ddp",
        default_root_dir=checkpoint_path,
    )
    trainer.fit(model, train_dataloader, val_dataloader)
