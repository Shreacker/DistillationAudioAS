import torch
from torch.utils.data import Dataset
import idx2numpy
import numpy as np

def collate_fn(items: list[dict]) -> dict[str, torch.Tensor]: #[{'image': array (28, 28), 'label': int}]
    data = {
        "image": np.stack([item["image"] for item in items], axis=0), # (32, 28, 28)                                                                             
        "label": np.stack([item["label"] for item in items], axis=0)  # (32, )                                                                                  
    }

    data = {
        "image": torch.tensor(data["image"]),
        "label": torch.tensor(data["label"])
    }

    return data


class MnistDataset(Dataset):
    def __init__(self, image_path: str, label_path: str):
        images = idx2numpy.convert_from_file(image_path).astype(np.float32) / 255.0 # np.array (60000, 28, 28)                                                                        
        labels = idx2numpy.convert_from_file(label_path).astype(np.int64) # np.array (60000,)

        self._data = [
            {
                "image": image, # (28, 28)
                "label": label
            }
            for image, label in zip(images, labels)
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        return self._data[idx]