import torch
from torch.utils.data import Dataset
import json
from torchvision import transforms
from PIL import Image
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
        self.base_path = base_path

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        image = Image.open(self.base_path + self.data_info[index]['file_name'])
        transformed_image = self.transforms(image)
        # condition: frequency information and geometric information
        condition = torch.tensor(self.data_info[index]['param'][:200][::4] + self.data_info[index]['param'][-4:])
        # label: S-parameters (this can be changed to other target values depend on the application scenario)
        label = torch.tensor(self.data_info[index]['param'][200:1000][::4])
        return transformed_image, condition, label