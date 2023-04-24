import torch
import datasets
import lightning.pytorch as pl
from torch.utils.data import Dataset
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CometLogger
import os
from torch import nn
from lightning.pytorch.callbacks import ModelCheckpoint
from all_datasets import SpeechDatasetLibre
from model import ASRLightningModule

def pad_collate(batch):
    ans = {
        'spectrograms': nn.utils.rnn.pad_sequence([x['spectrograms'].permute(1,0) for x in batch], batch_first=True).permute(0,2,1),
        'texts': [item['texts'] for item in batch]
    }
    return ans

hparams = {
    'dropout': 0.1,
    'rnn_input_size':256,
    'rnn_hidden_size':256,
    'rnn_num_layers':4,
    # 'cnn_num_layers':2,
    'fc2_hidden_size':512,
    'n_class':26+2,
    'global_seed':69,
    'batch_count_train' : 16,
    'batch_count_val' : 16,
    'learning_rate': 0.0008,
    'gamma': 0.9,
    # 'validate_each_n_train_batches': 500,
    # 'train_dataset': 'train-other-500',
    # 'test_dataset': 'test-other',
    # 'validate_each_n_train_batches': 5,
}

seed_everything(hparams['global_seed'])

# dataset_hugging = load_dataset("mozilla-foundation/common_voice_11_0", "ru", cache_dir="./data")
# dataset_train = CommonVoiceRu(dataset_hugging['train'], 'train') #CaptchaDataset(train_dir, 'train', char_set_pred, transform=train_transform, hparams=hparams)
# dataset_val = CommonVoiceRu(dataset_hugging['validation'], 'validation')

# dataset_hugging = load_dataset('bond005/sberdevices_golos_10h_crowd', cache_dir="./data")
# dataset_train = SpeechDataset(dataset_hugging['train'], 'train') #CaptchaDataset(train_dir, 'train', char_set_pred, transform=train_transform, hparams=hparams)
# dataset_val = SpeechDataset(dataset_hugging['validation'], 'validation')

# train_transform = nn.Sequential(
#     torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
#     # torchaudio.transforms.TimeMasking(time_mask_param=50, p=0.2)
#     )

dataset_train = SpeechDatasetLibre("train-clean-100", type_="train") #CaptchaDataset(train_dir, 'train', char_set_pred, transform=train_transform, hparams=hparams)
dataset_val = SpeechDatasetLibre("test-clean", type_="validation")

asr_model = ASRLightningModule(hparams)
# asr_model = ASRLightningModule.load_from_checkpoint(r'model.ckpt', hparams)

loader_kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=hparams['batch_count_train'], shuffle=True, collate_fn=pad_collate, **loader_kwargs)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=hparams['batch_count_val'], shuffle=False, collate_fn=pad_collate, **loader_kwargs)

comet_logger = CometLogger(
    api_key=os.environ.get("COMET_API_KEY"),
    save_dir="./comet",
    project_name="speech_recognition_pet_proj",
    experiment_name="4 rnn, 0.1 drop, common Voice",
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="WER/Validation",
    mode="min",
    dirpath="./models",
    filename="model-{epoch:02d}-{WER/Validation:.2f}",
    auto_insert_metric_name=False,
)

accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
trainer = pl.Trainer(max_epochs=500, 
                     accelerator=accelerator, 
                     logger=comet_logger, 
                    #  val_check_interval=hparams['validate_each_n_train_batches'], 
                     callbacks=[checkpoint_callback])
trainer.fit(model=asr_model, train_dataloaders=train_loader, val_dataloaders=val_loader)