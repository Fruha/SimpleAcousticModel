## Description
- Simple **acoustic model** for ASR
- Training model on **LibriSpeech** (WER~30%, CER~10%) or **CommonVoice**
- Automatic logging on server with **CometMl**

## Installation

```bash
git clone https://github.com/Fruha/SimpleAcousticModel
cd SimpleAcousticModel
pip install -r requirements.txt
```

## Usage
```bash
setx COMET_API_KEY "COMET_API_KEY"
python train.py
```