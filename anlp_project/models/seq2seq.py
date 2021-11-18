import pytorch_lightning as pl
import torch
from torch import nn, optim
from typing import Optional
from random import random


class EncoderRNN(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(input_size, self.config.hidden_size)
        self.gru = nn.GRU(self.config.hidden_size, self.config.hidden_size)

    def forward(self, input, hidden):
        emb = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(emb, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, config, output_size):
        super().__init__()
        self.config = config

        self.emb_layer_with_activation = nn.Sequential(
            nn.Embedding(output_size, self.config.hidden_size), nn.ReLU()
        )
        self.gru = nn.GRU(self.config.hidden_size, self.config.hidden_size)
        self.output_with_activation = nn.Sequential(
            nn.Linear(self.config.hidden_size, output_size), nn.LogSoftmax(dim=1)
        )

    def forward(self, input, hidden):
        emb = self.emb_layer_with_activation(input).view(1, 1, -1)
        output, hidden = self.gru(emb, hidden)
        output = self.output_with_activation(output[0])
        return output, hidden


class Seq2SeqRNN(pl.LightningModule):
    def __init__(self, config, input_size, output_size):
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size

        self.encoder = EncoderRNN(config, input_size)
        self.decoder = DecoderRNN(config, output_size)

        self.encoder_hidden = torch.zeros(
            1, 1, self.config.hidden_size, device=self.device
        )
        self.encoder_outputs: Optional[torch.tensor] = None

        # We need manual optimization because encoder optimizer is stepped
        # after we have done both the encoding/decoding step
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        enc_optim, dec_optim = self.optimizers()
        # both input and output are tensors
        # of shape (number of tokens in sentence, embedding size)
        input_tensor, output_tensor = batch
        input_word_count = input_tensor.size(0)
        output_word_count = output_tensor.size(0)

        self.encoder_outputs = torch.zeros(self.config.max_length, self.config.hidden_size, device=self.config.device)

        # an RNN works by iterating over all words one by one
        for word_index in range(input_word_count):
            encoder_output, self.encoder_hidden = self.encoder(
                input_tensor[word_index], self.encoder_hidden
            )
            self.encoder_outputs[word_index] = encoder_output[0, 0]

        decoder_input = torch.tensor([[self.config.bos_token]], device=self.device)
        decoder_hidden = self.encoder_hidden

        use_teacher_forcing = random() < self.config.teacher_forcing_ratio
        loss = 0

        loss_function = nn.NLLLoss()

        if use_teacher_forcing:
            for word_index in range(output_word_count):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                loss += loss_function(decoder_output, output_tensor[word_index])
                decoder_input = output_tensor[word_index]
        else:
            for word_index in range(output_word_count):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                loss += loss_function(decoder_output, output_tensor[word_index])
                if decoder_input.item() == self.config.eos_token:
                    break

        loss.backward()

        enc_optim.step()
        dec_optim.step()

        return loss.item() / output_word_count

    def configure_optimizers(self):
        enc_opt = optim.SGD(self.encoder.parameters(), self.config.lr)
        dec_opt = optim.SGD(self.decoder.parameters(), self.config.lr)

        return enc_opt, dec_opt
