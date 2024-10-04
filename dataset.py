from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

# Custom Dataset Class
class CoinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]