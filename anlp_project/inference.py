import logging

from anlp_project.config import Config
from anlp_project.datasets.europarl import EuroParl
from anlp_project.models.seq2seq import Seq2SeqRNN


def inference_model(config: Config):
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

    # -1 implies use all GPUs available
    gpu_count = -1 if torch.cuda.is_available() else 0
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=config.n_epochs,
        min_epochs=1,
        gpus=gpu_count,
        strategy="ddp",
    )
    # load the test dataset
    # load the trained model
    # run the trained model on inputs
    ...
