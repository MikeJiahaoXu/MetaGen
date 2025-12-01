import torch
from torch.utils.data import Dataset
import json
from torchvision import transforms
from PIL import Image
from pathlib import Path
class MetaDataset(Dataset):
    def __init__(self, meta_path, base_path, type='default'):
        data_info = []
        self.meta_path = meta_path
        with open(meta_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                data_info.append(data)
        self.data_info = data_info
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.base_path = Path(base_path)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        image_path = self.base_path / self.data_info[index]['file_name']
        image = Image.open(image_path).convert("L")
        transformed_image = self.transforms(image)
        condition = torch.tensor(self.data_info[index]['param'])
        return transformed_image, condition, self.data_info[index]['file_name']

