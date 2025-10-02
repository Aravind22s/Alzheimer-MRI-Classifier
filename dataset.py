# dataset.py
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class HuggingFaceMRIDataset(Dataset):
    """
    Wraps HuggingFace dataset into PyTorch Dataset
    """
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.num_classes = len(self.ds.features["label"].names)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        label = item["label"]

        img = img.convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        # repeat to 3 channels
        if img.shape[0] == 1:
            img = img.repeat(3,1,1)
        return img, label
