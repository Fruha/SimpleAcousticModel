import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl
from torchmetrics.functional import char_error_rate, word_error_rate
from utils import TextTransfromtaions





class Conv2DwithDropBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout):
        super(Conv2DwithDropBN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class LstmBlock(nn.Module):
    def __init__(self,rnn_input_size, rnn_hidden_size, dropout):
        super(LstmBlock, self).__init__()
        self.lstm = nn.LSTM(rnn_input_size ,hidden_size=rnn_hidden_size, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)
        self.post_proc = nn.Sequential(
             nn.LeakyReLU(),
             nn.BatchNorm1d(2*rnn_hidden_size),
        )
    def forward(self, x):
        # (Batch, Length, Features)
        x, _ = self.lstm(x)
        # (Batch, Length, Features)
        x = x.transpose(1,2)
        # (Batch, Features, Length)
        x = self.post_proc(x)
        x = x.transpose(1,2)
        # (Batch, Length, Features)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout):
        super(ResNetBlock, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # (batch, channel, feature, time)
        x_input = x
        x = F.leaky_relu(x)
        x = self.cnn1(x)
        x = self.batch_norm1(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)
        x = self.cnn2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x += x_input
        return x

class AcousticModel(nn.Module):
    def __init__(self, hparams):
        super(AcousticModel, self).__init__()
        self.cnn = nn.Sequential(
            Conv2DwithDropBN(1,32,3,1,hparams['dropout']),
            ResNetBlock(32,32,3,1,hparams['dropout']),
            Conv2DwithDropBN(32,32,3,(2,1),hparams['dropout']),
            ResNetBlock(32,32,3,1,hparams['dropout']),
            Conv2DwithDropBN(32,32,3,(2,2),hparams['dropout']),
            # ResNetBlock(32,32,3,1,hparams['dropout']),
            # ResNetBlock(32,32,3,1,hparams['dropout']),
        )
        self.fc1 = nn.Linear(1024, hparams['rnn_input_size'])
        self.rnn = nn.Sequential(
            *[LstmBlock(hparams['rnn_input_size'] if i == 0 else hparams['rnn_hidden_size']*2, hparams['rnn_hidden_size'], hparams['dropout']) for i in range(hparams['rnn_num_layers'])]
        )
        self.fc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hparams['rnn_hidden_size']*2,hparams['fc2_hidden_size']),
            # nn.Dropout(hparams['dropout']),
            nn.LeakyReLU(),
            nn.Linear(hparams['fc2_hidden_size'],hparams['n_class']),
        )
        

    def forward(self, x):
        x = x.unsqueeze(1)
        # (Batch, 1, N_mels, Length)
        x = self.cnn(x)
        # (Batch, Channels, Height, Length)
        x = x.transpose(1,3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.fc1(x)
        # (Batch, Length, Features)
        x = self.rnn(x)
        # (Batch, Length, Features)
        x = self.fc2(x)
        # (Batch, Length, N_class)
        return x

class ASRLightningModule(pl.LightningModule):
    def __init__(self, hparams:dict) -> None:
        super(ASRLightningModule,self).__init__()
        self.acoustic_model = AcousticModel(hparams=hparams)
        self.ctcloss = nn.CTCLoss()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        self.train_step_outputs = []
        self.train_step_y = []
        self.validation_step_outputs = []
        self.validation_step_y = []

    def training_step(self, batch, batch_idx):
        spectrograms, texts_target = batch['spectrograms'], batch['texts']
        lengths_target = torch.LongTensor([len(text) for text in texts_target])
        labels_target = [TextTransfromtaions.encode_str(text) for text in texts_target]
        labels_target = nn.utils.rnn.pad_sequence(labels_target, batch_first=True)
        output = self.acoustic_model(spectrograms)
        input_lengths = torch.full(size=(output.size(0),), fill_value=output.size(1), dtype=torch.long)
        logits = F.log_softmax(output, dim=2)
        loss = self.ctcloss(logits.permute(1,0,2), labels_target, input_lengths, lengths_target)

        pred_texts = [TextTransfromtaions.ctc_decoder(x) for x in output]
        self.log("CER/Train", char_error_rate(pred_texts, texts_target))
        self.log("WER/Train", word_error_rate(pred_texts, texts_target))

        self.log("Loss/Train", loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        spectrograms, texts_target = batch['spectrograms'], batch['texts']
        lengths_target = torch.LongTensor([len(text) for text in texts_target])
        labels_target = [TextTransfromtaions.encode_str(text) for text in texts_target]
        labels_target = nn.utils.rnn.pad_sequence(labels_target, batch_first=True)
        output = self.acoustic_model(spectrograms)
        input_lengths = torch.full(size=(output.size(0),), fill_value=output.size(1), dtype=torch.long)
        logits = F.log_softmax(output, dim=2)
        loss = self.ctcloss(logits.permute(1,0,2), labels_target, input_lengths, lengths_target)
        
        pred_texts = [TextTransfromtaions.ctc_decoder(x) for x in output]
        # print(pred_texts, texts_target)
        self.log("CER/Validation", char_error_rate(pred_texts, texts_target))
        self.log("WER/Validation", word_error_rate(pred_texts, texts_target))

        self.log("Loss/Validation", loss, prog_bar=True)

    def on_save_checkpoint(self, checkpoint):
        self.logger.experiment.log_model('model', 'models', overwrite=True)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.hparams['gamma'])
        return {'optimizer': self.optimizer, 'lr_scheduler':self.scheduler}