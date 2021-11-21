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

    # TODO: get checkpoint path from argument
    checkpoint_path = ''
    model = Seq2SeqRNN.load_from_checkpoint(checkpoint_path)
    logging.info('Parameters of loaded model: learning rate: %f', model.learning_rate)
    model.eval()
