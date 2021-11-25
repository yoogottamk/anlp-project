from argparse import Namespace
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


def get_args():
    return Namespace(
        activation_dropout=0.0,
        activation_fn="relu",
        adam_betas="(0.9, 0.98)",
        adam_eps=1e-08,
        adaptive_input=False,
        adaptive_softmax_cutoff=None,
        adaptive_softmax_dropout=0,
        arch="transformer_wmt_en_de",
        attention_dropout=0.0,
        best_checkpoint_metric="loss",
        bpe=None,
        bucket_cap_mb=25,
        clip_norm=0.0,
        cpu=False,
        criterion="label_smoothed_cross_entropy",
        cross_self_attention=False,
        curriculum=0,
        data="pre-processed-data-dir",
        dataset_impl=None,
        ddp_backend="c10d",
        decoder_attention_heads=8,
        decoder_embed_dim=512,
        decoder_embed_path=None,
        decoder_ffn_embed_dim=2048,
        decoder_input_dim=512,
        decoder_layers=12,
        decoder_learned_pos=False,
        decoder_normalize_before=False,
        decoder_output_dim=512,
        device_id=0,
        disable_validation=False,
        distributed_backend="nccl",
        distributed_init_method=None,
        distributed_no_spawn=False,
        distributed_port=-1,
        distributed_rank=0,
        distributed_world_size=1,
        dropout=0.3,
        empty_cache_freq=0,
        encoder_attention_heads=8,
        encoder_embed_dim=512,
        encoder_embed_path=None,
        encoder_ffn_embed_dim=2048,
        encoder_layers=12,
        encoder_learned_pos=False,
        encoder_normalize_before=False,
        fast_stat_sync=False,
        find_unused_parameters=False,
        fix_batches_to_gpus=False,
        fixed_validation_seed=None,
        fp16=False,
        fp16_init_scale=128,
        fp16_scale_tolerance=0.0,
        fp16_scale_window=None,
        gradient_as_delta=False,
        init_type="adaptive-profiling",
        keep_interval_updates=-1,
        keep_last_epochs=-1,
        label_smoothing=0.1,
        layer_wise_attention=False,
        lazy_load=False,
        left_pad_source="True",
        left_pad_target="False",
        load_alignments=False,
        log_format=None,
        log_interval=100,
        lr=[0.002],
        lr_scheduler="inverse_sqrt",
        max_epoch=0,
        max_sentences=None,
        max_sentences_valid=None,
        max_source_positions=1024,
        max_target_positions=1024,
        max_tokens=3584,
        max_tokens_valid=3584,
        max_update=50000,
        maximize_best_checkpoint_metric=False,
        memory_efficient_fp16=False,
        min_loss_scale=0.0001,
        min_lr=1e-09,
        mixed_precision=False,
        no_cross_attention=False,
        no_epoch_checkpoints=False,
        no_last_checkpoints=False,
        no_progress_bar=False,
        no_save=False,
        no_save_optimizer_state=False,
        no_token_positional_embeddings=False,
        num_workers=1,
        optimizer="adam",
        optimizer_overrides="{}",
        plot_gradient=False,
        plot_stability=False,
        plot_variance=False,
        raw_text=False,
        required_batch_size_multiple=8,
        reset_dataloader=False,
        reset_lr_scheduler=False,
        reset_meters=False,
        reset_optimizer=False,
        restore_file="checkpoint_last.pt",
        save_dir="model-save-dir",
        save_interval=1,
        save_interval_updates=0,
        seed=1,
        sentence_avg=False,
        share_all_embeddings=True,
        share_decoder_input_output_embed=False,
        share_layer_num=2,
        share_params_cross_layer=True,
        share_type="cycle_reverse",
        skip_invalid_size_inputs_valid_test=False,
        source_lang=None,
        target_lang=None,
        task="translation",
        tbmf_wrapper=False,
        tensorboard_logdir="",
        threshold_loss_scale=None,
        tokenizer=None,
        train_subset="train",
        update_freq=[32],
        upsample_primary=1,
        use_bmuf=False,
        user_dir=None,
        valid_subset="valid",
        validate_interval=1,
        warmup_init_lr=1e-07,
        warmup_updates=8000,
        weight_decay=0.0,
    )


def get_transformer_model(args=None):
    if not args:
        args = {}
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    model = TransformerModel(args, encoder, decoder)
    return model

# return source and target dictionaries
def get_dictionaries(dataset: EuroParl):
    de_dict = Dictionary()

def train_model(config: Config):
    dataset = EuroParl(config=config)
    args = get_args()
    de_dict, en_dict = get_dictionaries(dataset)
    task = TranslationTask(args, de_dict, en_dict)
    model = get_transformer_model(args)

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
