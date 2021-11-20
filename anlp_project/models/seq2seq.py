import pytorch_lightning as pl
import torch
from torch import nn, optim
from anlp_project.config import Config
from random import random


class EncoderRNN(nn.Module):
    def __init__(self, config: Config, input_size: int):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(input_size, self.config.hidden_size)
        self.gru = nn.GRU(self.config.hidden_size, self.config.hidden_size)

    def forward(self, input, hidden):
        # Input is a 1-hot tensor, we need to get the index
        # so as to pass it to embedding
        # TODO: change to axis=1 for batch training
        input = input.argmax(axis=0)
        emb_i = self.embedding(input)
        emb = emb_i.view(1, 1, -1)
        output, hidden = self.gru(emb, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, config: Config, output_size: int):
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
        # TODO: change to axis=1 for batch training
        input = input.argmax(axis=0)
        emb = self.emb_layer_with_activation(input).view(1, 1, -1)
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
        # both input and output are tensors
        # of shape (number of tokens in sentence, embedding size)
        input_tensor, target_output_tensor = batch
        # TODO: change for batched training
        input_tensor = input_tensor[0]
        target_output_tensor = target_output_tensor[0]
        input_word_count = input_tensor.size(0)
        target_word_count = target_output_tensor.size(0)

        encoder_hidden = torch.zeros(
            1, 1, self.config.hidden_size, device=self.device
        )
        encoder_outputs = torch.zeros(self.config.max_length, self.config.hidden_size, device=self.device)

        # an RNN works by iterating over all words one by one
        for word_index in range(input_word_count):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[word_index], encoder_hidden
            )
            encoder_outputs[word_index] = encoder_output[0, 0]  # TODO: what is the meaning of [0, 0]?

        # input_tensor[0] is the bos token
        # TODO: convert to axis=1 for batched training
        decoder_input = torch.tensor([[input_tensor[0].argmax(axis=0)]], device=self.device)
        decoder_hidden = encoder_hidden
        return decoder_input, decoder_hidden, target_output_tensor, target_word_count

    def training_step(self, batch, batch_idx):
        enc_optim, dec_optim = self.optimizers()
        decoder_input, decoder_hidden, target_tensor, target_word_count = self.move_encoder_forward(batch)

        use_teacher_forcing = random() < self.config.teacher_forcing_ratio
        loss = 0

        loss_function = nn.NLLLoss()

        if use_teacher_forcing:
            for word_index in range(target_word_count):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                loss += loss_function(decoder_output, target_tensor[word_index])
                decoder_input = target_tensor[word_index]
        else:
            for word_index in range(target_word_count):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                # what does this do?
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                loss += loss_function(decoder_output, target_tensor[word_index])

                if decoder_input.item() == self.config.eos_token:
                    # breaking early: aren't we helping the loss to be low?
                    # because in the end we're diving by target_word_count
                    # instead of the word count we actually outputted
                    break

        loss.backward()

        enc_optim.step()
        dec_optim.step()

        return loss.item() / target_word_count

    def configure_optimizers(self):
        enc_opt = optim.SGD(self.encoder.parameters(), self.config.lr)
        dec_opt = optim.SGD(self.decoder.parameters(), self.config.lr)

        return enc_opt, dec_opt

    def validation_step(self, batch, batch_idx):
        # we don't want to train/backprop
        with torch.no_grad():
            decoder_input, decoder_hidden, target_tensor, target_word_count = self.move_encoder_forward(batch)

            loss = 0

            loss_function = nn.NLLLoss()

            for word_index in range(target_word_count):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                # TODO: batch training
                # NLLLoss expects NXC tensor as the source and (N,) shape tensor for target
                target_class = target_tensor[word_index].argmax(axis=0).item()
                target_class = torch.LongTensor([target_class])
                loss_output = loss_function(decoder_output, target_class)
                loss += loss_output
                if decoder_input.item() == self.config.eos_token:
                    break

            return loss.item() / target_word_count
