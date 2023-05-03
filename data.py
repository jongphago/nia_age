import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, filepath_list, transform=None):
        self.images = []
        self.labels = []
        self.files = []
        for filepath in filepath_list:
            basename = filepath
            self.files.append(os.path.basename(filepath))
            # self.labels.append(int(basename[4:6]))
            # age_over = int(basename[0:2])
            age = os.path.basename(filepath).split("_")[0]

            # self.labels.append(int(basename[0:2]))
            # print("Age--", age)
            age = int(age)
            if age < 0:
                age = -age
                # print("age---", age)

            self.labels.append(np.int(age))

            # if int(age) < 0:
            # print("basename----", os.path.basename(filepath))
            # print("Age is minus----", -age)

            # else:
            # continue

            # print(int(basename[0:2]))
            img = np.array(Image.open(filepath).convert("RGB"))
            self.images.append(img)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        img = self.images[index]
        # img = self.images[index].astype(np.float32)
        label = self.labels[index]
        files = self.files[index]
        if self.transform:
            img = self.transform(img)
        sample = {"image": img, "label": label, "file": files}
        return sample


class NiaDataset(Dataset):
    RANGE_TO_MEDIAN = {
        "a": (1 + 6) / 2,
        "b": (7 + 12) / 2,
        "c": (13 + 19) / 2,
        "d": (20 + 30) / 2,
        "e": (31 + 45) / 2,
        "f": (46 + 55) / 2,
        "g": (56 + 66) / 2,
        "h": (67 + 80) / 2,
        "above": 90,
    }
    AGE_GROUPS = ["a", "b", "c", "d", "e", "f", "g", "h", "above"]

    @staticmethod
    def age_to_age_groups(age):
        if age <= 6:
            return "a"
        if age <= 12:
            return "b"
        if age <= 19:
            return "c"
        if age <= 30:
            return "d"
        if age <= 45:
            return "e"
        if age <= 55:
            return "f"
        if age <= 66:
            return "g"
        if age <= 80:
            return "h"
        return "above"

    def __init__(self, meta_npy_path: str, transform=None):
        super(NiaDataset, self).__init__()
        self.data_list = np.load(meta_npy_path, allow_pickle=True)
        self.transform = transform

    def __getitem__(self, idx: int):
        datum = self.data_list[idx]
        img_path = datum["img_path"]
        img = np.array(Image.open(img_path).convert("RGB"))
        files = os.path.basename(img_path)
        infos = files.split("_")
        if datum["age"] is None:
            data_type = "age"
            age = int(self.RANGE_TO_MEDIAN[datum["age_class"]])
            age_class = datum["age_class"]
        else:
            data_type = "kinship"
            age = datum["age"]
            age_class = self.age_to_age_groups(age)
        age_class = self.AGE_GROUPS.index(age_class)

        if self.transform:
            img = self.transform(img)
        sample = {
            "image": img,
            "age": age,
            "age_class": age_class,
            "file": files,
            "data_type": data_type,
            "family_id": infos[0],
            "personal_id": f"{infos[0]}-{infos[2]}",
        }
        return sample

    def __len__(self) -> int:
        return len(self.data_list)
