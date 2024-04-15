import os
import tarfile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import io
from torch.utils.data import DataLoader

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """
    
    def __init__(self, tar_path):
        self.tar_path = tar_path
        with tarfile.open(self.tar_path, "r") as tar:
            self.filenames = [member for member in tar.getmembers() if member.isfile()]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with tarfile.open(self.tar_path, "r") as tar:
            member = self.filenames[idx]
            file = tar.extractfile(member)
            image = Image.open(io.BytesIO(file.read()))
        image = self.transform(image)
        label = int(member.name.split('_')[1].split('.')[0])
        return image, label

if __name__ == '__main__':
    train_dataset = MNIST('../data/train.tar')
    test_dataset = MNIST('../data/test.tar')
    
    train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=64, shuffle=True)

    print("Label of the first test image:", test_dataset[0][1])
    print("Label of the first train image:", train_dataset[0][1])
    print("Number of training images:", len(train_dataset))
    print("Number of testing images:", len(test_dataset))
    train_images, train_labels = next(iter(train_data))
    print("Shape of the first batch of train images:", train_images.shape)
    test_images, test_labels = next(iter(test_data))
    print("Shape of the first batch of test images:", test_images.shape)
