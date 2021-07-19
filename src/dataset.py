import torch
from torch.utils.data import Dataset
import cv2

class CustomDataset(Dataset):

    def __init__(self, df: pd.DataFrame, augmentation = None):
        self.df = df
        self.file_name = df.file_name.values
        self.labels = df.labels.values
        self.augmentation = augmentation


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        file_name = self.file_name[idx]
        label_name = self.labels[idx]

        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.augmentation is None:
            augmented = self.augmentation(image = image)
            image = augmented["image"]


        """
        if loss function is BCEWithLogitsLoss, input data into loss function must be torch Float type 

        """
        return {
            "image" : torch.tensor(image, dtype = float), 
            "label": torch.tensor(label_name, dtype = float), 
        }