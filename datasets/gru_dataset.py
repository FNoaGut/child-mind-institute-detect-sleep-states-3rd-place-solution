import torch
from torch.utils.data import Dataset

def preprocess(numerical_array,
               mask_array
               ):
    attention_mask = mask_array == 0

    return {
        'input_data_numerical_array': numerical_array,
        'input_data_mask_array': mask_array,
        'attention_mask': attention_mask
    }


class CustomDataset(Dataset):
    def __init__(self, numerical_array,
                 mask_array,
                 train=True, y=None):
        self.numerical_array = numerical_array
        self.mask_array = mask_array
        self.train = train
        self.y = y

    def __len__(self):
        return len(self.numerical_array)

    @staticmethod
    def batch_to_device(batch, device):
        input_data_numerical_array = batch['input_data_numerical_array'].to(device)
        input_data_mask_array = batch['input_data_mask_array'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target = batch["y"].to(device)
        return input_data_numerical_array, input_data_mask_array, attention_mask, target

    def __getitem__(self, item):
        data = preprocess(
            self.numerical_array[item],
            self.mask_array[item]
        )

        # Return the processed data where the lists are converted to `torch.tensor`s
        if self.train:
            return {
                'input_data_numerical_array': torch.tensor(data['input_data_numerical_array'], dtype=torch.float32),
                'input_data_mask_array': torch.tensor(data['input_data_mask_array'], dtype=torch.long),
                'attention_mask': torch.tensor(data["attention_mask"], dtype=torch.bool),
                "y": torch.tensor(self.y[item], dtype=torch.float32)
            }
        else:
            return {
                'input_data_numerical_array': torch.tensor(data['input_data_numerical_array'], dtype=torch.float32),
                'input_data_mask_array': torch.tensor(data['input_data_mask_array'], dtype=torch.long),
                'attention_mask': torch.tensor(data["attention_mask"], dtype=torch.bool),
            }
