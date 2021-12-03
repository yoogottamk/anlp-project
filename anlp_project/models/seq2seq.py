import logging
from random import random
from typing import Tuple, List

import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F

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


class AttentionDecoderRNN(nn.Module):
    def __init__(
        self,
        config: Config,
        output_size: int,
    ):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            output_size, self.config.hidden_size, padding_idx=0
        )

        self.attn = nn.Linear(self.config.hidden_size * 2, self.config.max_length)
        self.attn_combine = nn.Linear(
            self.config.hidden_size * 2, self.config.hidden_size
        )
        self.dropout = nn.Dropout(self.config.dropout)

        self.gru = nn.GRU(self.config.hidden_size, self.config.hidden_size)
        self.output_with_activation = nn.Sequential(
            nn.Linear(self.config.hidden_size, output_size), nn.LogSoftmax(dim=1)
        )

    def forward(self, input, hidden, encoder_outputs):
        batch_size = input.size(0)
        emb = self.embedding(input).view(1, batch_size, -1)
        emb = self.dropout(emb)

        #                                  shape is (batch, 2 * hidden_size)
        #                                  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #                        shape is  (batch, sentence max length)
        #                        vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        attn_weights = F.softmax(self.attn(torch.cat((emb[0], hidden[0]), 1)), dim=1)
        #              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #              shape is (batch, max length of sentence)

        # encoder_outputs = (batch, max length of sentence, hidden size)
        # attn_weights: (batch, 1, max length of sentence)
        attn_weights = attn_weights.unsqueeze(1)

        # bmm == batch matrix matrix product
        attn_applied = torch.bmm(attn_weights, encoder_outputs).view(1, batch_size, -1)

        output = torch.cat((emb[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        # output is (1, BATCH_SIZE, HIDDEN_SIZE)
        output, hidden = self.gru(output, hidden)

        output = self.output_with_activation(output[0])
        return output, hidden, attn_weights


class Seq2SeqRNN(pl.LightningModule):
    def __init__(self, config: Config, input_size: int, output_size: int):
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size

        self.encoder = EncoderRNN(config, input_size)
        self.decoder = AttentionDecoderRNN(config, output_size)

        # We need manual optimization because encoder optimizer is stepped
        # after we have done both the encoding/decoding step
        self.automatic_optimization = False

    def _move_encoder_forward(self, batch: Tuple[torch.LongTensor, torch.LongTensor]):
        input_tensor, target_output_tensor = batch[0], batch[1]
        batch_size = input_tensor.size(0)
        # first token of first sentence in the batch
        bos_token = input_tensor[0, 0]
        # sanity check
        assert bos_token == 2, f"Value of bos token in w2i_en is {bos_token}"

        # inputs and targets have been padded
        word_count = input_tensor.size(1)

        encoder_hidden = torch.zeros(
            1, batch_size, self.config.hidden_size, device=self.device
        )
        # (batchsize, sentence max length, attention vector size (hidden size))
        encoder_outputs = torch.zeros(
            batch_size,
            self.config.max_length,
            self.config.hidden_size,
            device=self.device,
        )

        # an RNN works by iterating over all words one by one
        for word_index in range(word_count):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[:, word_index], encoder_hidden
            )
            # shape of encoder_output is (1, batchsize, hiddensize)
            encoder_outputs[:, word_index, :] = encoder_output[0, :, :]

        decoder_input = torch.full((batch_size, 1), bos_token, device=self.device)
        decoder_hidden = encoder_hidden
        return (
            decoder_input,
            decoder_hidden,
            target_output_tensor,
            self.config.max_length,
            encoder_outputs,
        )

    def _move_decoder_forward(
        self,
        decoder_input,
        decoder_hidden,
        target_tensor,
        target_word_count,
        encoder_outputs,
    ):
        loss_function = nn.NLLLoss()
        loss = 0

        for word_index in range(target_word_count):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # decoder output is of shape (batch_size, outputsize)
            topv, topi = decoder_output[
                :,
            ].topk(1)
            decoder_input = topi.squeeze().unsqueeze(1).detach()
            # now decoder input is of shape (batch_size, 1)

            # TODO: ignore loss from PAD

            # NLLLoss expects NXC tensor as the source and (N,) shape tensor for target
            loss += loss_function(decoder_output, target_tensor[:, word_index])

        return loss

    def training_step(self, batch, _batch_idx):
        enc_optim, dec_optim = self.optimizers()
        (
            decoder_input,
            decoder_hidden,
            target_tensor,
            target_word_count,
            encoder_outputs,
        ) = self._move_encoder_forward(batch)

        use_teacher_forcing = random() < self.config.teacher_forcing_ratio
        loss = 0

        if use_teacher_forcing:
            loss_function = nn.NLLLoss()
            for word_index in range(target_word_count):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                loss += loss_function(decoder_output, target_tensor[:, word_index])
                decoder_input = target_tensor[:, word_index].unsqueeze(1)
        else:
            loss = self._move_decoder_forward(
                decoder_input,
                decoder_hidden,
                target_tensor,
                target_word_count,
                encoder_outputs,
            )

        loss.backward()

        enc_optim.step()
        dec_optim.step()

        train_loss = loss.item() / target_word_count

        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"train_loss": train_loss}

    def configure_optimizers(self):
        enc_opt = optim.Adam(self.encoder.parameters(), self.config.lr)
        dec_opt = optim.Adam(self.decoder.parameters(), self.config.lr)

        return enc_opt, dec_opt

    def validation_step(self, batch, batch_idx):
        (
            decoder_input,
            decoder_hidden,
            target_tensor,
            target_word_count,
            encoder_outputs,
        ) = self._move_encoder_forward(batch)

        loss = self._move_decoder_forward(
            decoder_input,
            decoder_hidden,
            target_tensor,
            target_word_count,
            encoder_outputs,
        )

        val_loss = loss.item() / target_word_count

        self.log("validation_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        return val_loss

    def evaluate(self, input_sentence: List[int]):
        # one batch of the input sentence
        input_tensor = torch.LongTensor([input_sentence])
        # just passing input_tensor in target tensor too
        # as it doesn't matter anyway (we're not computing loss)
        batch = (input_tensor, input_tensor)
        (
            decoder_input,
            decoder_hidden,
            target_tensor,
            target_word_count,
            encoder_outputs,
        ) = self._move_encoder_forward(batch)

        decoded_words = []

        for word_index in range(self.config.max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output[:,].topk(1)
            if topi.item() == self.config.eos_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(topi.item())

            decoder_input = torch.LongTensor([topi])

        return decoded_words
