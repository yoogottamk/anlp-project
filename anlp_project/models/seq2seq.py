import pytorch_lightning as pl
import torch
from torch import nn, optim


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

        self.emb_layer = nn.Sequential(
            nn.Embedding(output_size, self.config.hidden_size), nn.ReLU()
        )
        self.gru = nn.GRU(self.config.hidden_size, self.config.hidden_size)
        self.output = nn.Sequential(
            nn.Linear(self.config.hidden_size, output_size), nn.LogSoftmax(dim=1)
        )

    def forward(self, input, hidden):
        emb = self.emb_layer(input).view(1, 1, -1)
        output, hidden = self.gru(emb, hidden)
        return self.output(output)


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

    def training_step(self, batch, batch_idx, optim_idx):
        if optim_idx == 0:
            # encoder step
            pass
        elif optim_idx == 1:
            # decoder step
            pass
        else:
            # error?
            raise ValueError("New optim added but training_step not configured")

    def configure_optimizers(self):
        enc_opt = optim.SGD(self.encoder.parameters(), self.config.lr)
        dec_opt = optim.SGD(self.decoder.parameters(), self.config.lr)

        return enc_opt, dec_opt
