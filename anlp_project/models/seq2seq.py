from random import random

import pytorch_lightning as pl
import torch
from torch import nn, optim

from anlp_project.config import Config


class EncoderRNN(nn.Module):
    def __init__(self, config: Config, input_size: int):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            input_size, self.config.hidden_size, padding_idx=0
        )
        self.gru = nn.GRU(self.config.hidden_size, self.config.hidden_size)

    def forward(self, input, hidden):
        emb_i = self.embedding(input)
        # GRU expects L, N, H
        # since we're passing a single word, L=1
        # N = batch size
        # H = hidden_state size for us
        emb = emb_i.view(1, input.size(0), -1)
        output, hidden = self.gru(emb, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, config: Config, output_size: int):
        super().__init__()
        self.config = config

        self.emb_layer_with_activation = nn.Sequential(
            nn.Embedding(output_size, self.config.hidden_size, padding_idx=0), nn.ReLU()
        )
        self.gru = nn.GRU(self.config.hidden_size, self.config.hidden_size)
        self.output_with_activation = nn.Sequential(
            nn.Linear(self.config.hidden_size, output_size), nn.LogSoftmax(dim=1)
        )

    def forward(self, input, hidden):
        emb = self.emb_layer_with_activation(input).view(1, input.size(0), -1)
        output, hidden = self.gru(emb, hidden)
        output = self.output_with_activation(output[0])
        return output, hidden


class Seq2SeqRNN(pl.LightningModule):
    def __init__(self, config: Config, input_size: int, output_size: int):
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size

        self.encoder = EncoderRNN(config, input_size)
        self.decoder = DecoderRNN(config, output_size)

        # We need manual optimization because encoder optimizer is stepped
        # after we have done both the encoding/decoding step
        self.automatic_optimization = False

    def move_encoder_forward(self, batch):
        input_tensor, target_output_tensor = batch[0], batch[1]
        batch_size = input_tensor.size(0)
        # first token of first sentence in the batch
        bos_token = input_tensor[0, 0]
        # sanity check
        assert bos_token == 1, "What is the value of bos token in w2i_en?"

        # inputs and targets have been padded
        wc = input_tensor.size(1)

        encoder_hidden = torch.zeros(
            1, batch_size, self.config.hidden_size, device=self.device
        )
        encoder_outputs = torch.zeros(
            self.config.max_length, self.config.hidden_size, device=self.device
        )

        # an RNN works by iterating over all words one by one
        for word_index in range(wc):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[:, word_index], encoder_hidden
            )
            encoder_outputs[word_index] = encoder_output[
                0, 0
            ]  # TODO: why exactly do we need [0, 0] here?

        decoder_input = torch.full(
            (self.config.batch_size, 1), bos_token, device=self.device
        )
        decoder_hidden = encoder_hidden
        return (
            decoder_input,
            decoder_hidden,
            target_output_tensor,
            self.config.max_length,
        )

    def _move_decoder_forward(
        self, decoder_input, decoder_hidden, target_tensor, target_word_count
    ):
        loss_function = nn.NLLLoss()
        loss = 0

        for word_index in range(target_word_count):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # what does this do?
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            # NLLLoss expects NXC tensor as the source and (N,) shape tensor for target
            loss += loss_function(decoder_output, target_tensor[:, word_index])

            if decoder_input.item() == self.config.eos_token:
                # breaking early: aren't we helping the loss to be low?
                # because in the end we're diving by target_word_count
                # instead of the word count we actually outputted
                break

        return loss

    def training_step(self, batch, _batch_idx):
        enc_optim, dec_optim = self.optimizers()
        (
            decoder_input,
            decoder_hidden,
            target_tensor,
            target_word_count,
        ) = self.move_encoder_forward(batch)

        use_teacher_forcing = random() < self.config.teacher_forcing_ratio
        loss = 0

        if use_teacher_forcing:
            loss_function = nn.NLLLoss()
            for word_index in range(target_word_count):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden
                )
                decoder_input = target_tensor[word_index]
                loss += loss_function(decoder_output, target_tensor[word_index])
        else:
            loss = self._move_decoder_forward(
                decoder_input, decoder_hidden, target_tensor, target_word_count
            )

        loss.backward()

        enc_optim.step()
        dec_optim.step()

        return {"train_loss": loss.item() / target_word_count}

    def configure_optimizers(self):
        enc_opt = optim.SGD(self.encoder.parameters(), self.config.lr)
        dec_opt = optim.SGD(self.decoder.parameters(), self.config.lr)

        return enc_opt, dec_opt

    def validation_step(self, batch, batch_idx):
        # we don't want to train/backprop
        with torch.no_grad():
            (
                decoder_input,
                decoder_hidden,
                target_tensor,
                target_word_count,
            ) = self.move_encoder_forward(batch)

            loss = self._move_decoder_forward(
                decoder_input, decoder_hidden, target_tensor, target_word_count
            )

            return loss.item() / target_word_count
