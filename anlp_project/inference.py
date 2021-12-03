import logging

import torch

from anlp_project.config import Config
from anlp_project.datasets.europarl import EuroParl
from anlp_project.models.seq2seq import Seq2SeqRNN


def inference_model(config: Config, checkpoint_file: str, input_sentence: str):
    if not input_sentence:
        # parliament related sample sentence
        # it is German for: "Our citizens need better water supply to their house"
        input_sentence = "unsere burger brauchen eine bessere wasserversorgung ihres hauses"

    dataset = EuroParl(config=config)

    # input is English, output is German
    input_size = dataset.de_vocab_size
    output_size = dataset.en_vocab_size

    logging.info(
        "Input size (German vocab): %d; Output size (English vocab): %d",
        input_size,
        output_size,
    )

    model = Seq2SeqRNN(config, input_size, output_size)
    model.load_state_dict(torch.load(checkpoint_file)["state_dict"])
    model.eval()

    logging.info(
        "Parameters of loaded model: input/output size: %d/%d",
        model.input_size,
        model.output_size,
    )

    token_sentence = dataset.sentence_to_indices(input_sentence)
    token_sentence = [dataset.BOS_TOKEN_INDEX, *token_sentence, dataset.EOS_TOKEN_INDEX]
    decoded_words_tokens = model.evaluate(token_sentence)
    # convert token integers back to words
    decoded_sentence = dataset.indices_to_sentence(decoded_words_tokens)
    print(decoded_sentence)
