from comet_ml import API
import os
import string
import torch
import re


def load_model(workspace:str, project_name:str, experiment_key:str, model_key:str, model_output_path: str = 'model.ckpt') -> None:
    """
    Load model from assets in experiment
    Arguments:
        workspace: workspace or account name
        project_name: project name
        experiment_key: experiment id
        model_key: model id
        model_output_path: path for saving model
    """
    api = API(api_key=os.environ.get("COMET_API_KEY"))
    experiment = api.get(f'{workspace}/{project_name}/{experiment_key}')
    asset_response = experiment.get_asset(
        model_key,
        return_type="response",
        stream=True,
    )
    with open(model_output_path, 'wb') as fd:
        for chunk in asset_response.iter_content(chunk_size=1024*1024):
            fd.write(chunk)

class TextTransfromtaions:
    # alplabet = "' абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    alplabet = "' " + string.ascii_lowercase
    char_to_int = {char:i for i,char in enumerate(alplabet)}
    int_to_char = {i:char for i,char in enumerate(alplabet)}

    @classmethod
    def encode_str(self, str_:str)-> torch.LongTensor:
        """
        Encode string by label encoding
        """
        str_ = str_.lower()
        ans = [self.char_to_int[char] for char in str_]
        return torch.LongTensor(ans)

    @classmethod
    def decode_str(self, labels)-> str:
        """
        Decode label array to string
        """
        ans = ''.join([self.int_to_char[int(i)] for i in labels])
        return ans

    @classmethod
    def ctc_decoder(self, data: torch.Tensor)-> str:
        """
        Decode CTC output
        Arguments:
            data: (L,C)
            char_set_pred: str of all chars
        Return:
            Tensor if char_set_pred is None\n
            Str if char_set_pred is not None
        """
        data = data.argmax(1)
        data = torch.unique_consecutive(data)
        data = data[data != 0]
        data = self.decode_str(data.cpu().numpy())
        return data

    @classmethod
    def preprocess_str(self, input_str:str):
        """
        Preprocessing string to lower and keep only chars
        """
        output_str = input_str.lower()
        output_str = re.sub(r'[^a-z\s]', '', output_str)
        # output_str = re.sub(r'[^а-я\s]', '', output_str)
        output_str = re.sub(r'\s+', ' ', output_str)
        output_str = output_str.strip()
        return output_str