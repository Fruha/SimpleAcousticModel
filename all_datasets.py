import torch
import torchaudio
from torch.utils.data import Dataset
import os
from utils import TextTransfromtaions

class CommonVoiceRu(Dataset):
    def __init__(self, dataset_hugging, type_:str, transform = None):
        self.dataset_hugging = dataset_hugging
        self.transform = transform
        self.type_ = type_
        
    def __getitem__(self, idx):
        item = self.dataset_hugging[idx]
        sample_rate = item['audio']['sampling_rate']
        data = torch.Tensor(item['audio']['array'])
        to_mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=128, n_fft=sample_rate//40) #normalized =True
        mel_spec = to_mel_spec(data)
        if self.transform:
            mel_spec = self.transform(mel_spec)
        if self.type_ == 'train' or self.type_ == 'validation':
            ans = {
                'spectrograms': mel_spec,
                'texts': TextTransfromtaions.preprocess_str(item['sentence'])
            }
        elif self.type_ == 'test':
            ans = {
                'spectrograms' : mel_spec
            }
        else:
            raise Exception('Not valid type of dataset')
        return ans

    def __len__(self):
        return len(self.dataset_hugging)


class SpeechDatasetLibre(Dataset):
    def __init__(self, url:str, type_:str, transform = None, cache_dir='./data'):
        os.makedirs(cache_dir, exist_ok=True)
        self.dataset = torchaudio.datasets.LIBRISPEECH(cache_dir, url=url, download=True)
        self.transform = transform
        self.type_ = type_

    def __getitem__(self, idx):
        item = self.dataset[idx]
        data = item[0].squeeze()
        sample_rate = item[1]
        to_mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=128, n_fft=sample_rate//40) #normalized =True
        mel_spec = to_mel_spec(data)
        if self.transform:
            mel_spec = self.transform(mel_spec)
        if self.type_ == 'train' or self.type_ == 'validation':
            ans = {
                'spectrograms': mel_spec,
                'texts': item[2].lower()
            }
        elif self.type_ == 'test':
            ans = {
                'spectrograms' : mel_spec
            }
        else:
            raise Exception('Not valid type of dataset')
        return ans

    def __len__(self):
        return len(self.dataset)