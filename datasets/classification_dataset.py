from torchvision.transforms.v2 import Transform
import torchvision.transforms.v2 as transforms
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
from typing import *
from pandas import DataFrame
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing


def get_cpu_count() -> int:
    """Get the number of CPUs available in the system"""
    return multiprocessing.cpu_count()


class DermaClassificationDataset(Dataset):
    """This is a dataset for classifying skin lesions"""

    def __init__(
        self,
        root_img_folder: Path,
        image_dataframe: DataFrame,
        transform: Transform = None,
    ) -> None:
        if not DermaClassificationDataset._is_valid_structure(image_dataframe):
            raise Exception(
                f"The provided dataframe does not have the correct structure. Check the names of the columns"
            )
        self.root_img_folder: Path = root_img_folder
        """The base folder to look for the images in the dataframe"""
        self.image_df: DataFrame = image_dataframe
        """This is a dataframe containing information about the skin image lesions
        for classification"""
        self.transform: Transform = transform
        """The transformation to apply to images"""

    def __len__(self) -> int:
        return len(self.image_df)

    def __getitem__(self, index: int) -> tuple[Tensor, str, str]:
        # TODO: Implement this
        relative_path_str, metaclass = self.image_df.iloc[
            index,
            [0, 2],
        ]
        image_path = self.root_img_folder / relative_path_str
        image = Image.open(str(image_path))
        if self.transform:
            image = self.transform(image)
        else:
            def_transform = transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(dtype=torch.float32, scale=True),
                ]
            )
            image = def_transform(image)
        label = self.metaclass_to_int(metaclass)
        return image, label

    @staticmethod
    def collate_fn(batch):
        """The collate function to use for the dataloaders of this dataset"""
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        return images, labels

    def get_balanced_dataloader(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        """Get a balanced dataloader for iterating over this dataset"""
        class_weights = self.get_categories_and_totals()
        sample_weights = [0] * len(self)
        num_workers = min(max(0, num_workers), get_cpu_count())

        # Instead of iterating over the dataset, we can use the labels directly from the dataframe
        series = self.image_df["metaclass"]
        for idx, label in tqdm(enumerate(series), total=len(series), colour="green"):
            label_num = self.metaclass_to_int(label)
            class_weight = class_weights[label_num]
            sample_weights[idx] = 1 / class_weight

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        loader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
        )
        return loader

    def get_basic_dataloader(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        num_workers = min(max(0, num_workers), get_cpu_count())
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def _is_valid_structure(image_dataframe: DataFrame):
        """Private method to check if the dataframe has the correct structure"""
        columns = ["image_path", "classification", "metaclass"]
        # TODO: Check the types (Not that important)
        return all(column in image_dataframe.columns for column in columns)

    def get_all_images_meta_cat(self, category: str) -> DataFrame:
        """Get all the rows of images belonging to a certain metaclass category"""
        return self.image_df[self.image_df["metaclass"] == category]

    def get_meta_labels_of_df(self) -> list[str]:
        """Get all the labels of the metaclass"""
        return self.image_df["metaclass"].unique().tolist()

    def get_labels(self) -> list[str]:
        """Returns the expected meta-labels of the dataset. The list is sorted in ascending order"""
        return sorted(["MEL", "BCC", "SCC", "OTHER"])

    def get_real_label_in_df(self) -> list[str]:
        """Get all the real classification labels."""
        return self.image_df["classification"].unique().tolist()

    def metaclass_to_int(self, label: str) -> int:
        """Returns the associated"""
        labels = self.get_labels()
        num = 0
        try:
            num = int(labels.index(label))
        except:
            raise Exception(
                f"It seems that the label '{label}' is not a valid label of the dataset. Please, check the case of the label it is all with UpperCase"
            )
        return num

    def int_to_metaclass(self, label_index: int):
        """Given an integer it returns the associated classification"""
        labels = self.get_labels()
        labels_length = len(labels)
        if not (0 <= label_index < labels_length):
            raise Exception(
                f"The valid labels are between 0 and {labels_length - 1}, but you are asking for label {label_index}"
            )
        return labels[label_index]

    def get_name_map(self):
        """Returns a dictionary that maps a label name to a full descriptive name.
        Example: MEL its translated to Melanoma"""
        return {
            "MEL": "Melanoma",
            "BCC": "Basal Cell Carcinoma",
            "SCC": "Squamous Cell Carcinoma",
            "OTHER": "Other",
        }

    def get_categories_and_totals(self) -> dict[int, int]:
        """
            Returns a dictionary where the key is the integer representing the category
        and the value is the number of images in that category.

        Returns:
            dict: A dictionary where each key is a category (as an integer) and the value is the total count of images in that category.
        """
        # Get the DataFrame of images
        df = self.image_df
        # Count the occurrences of each category in the "metaclass" column
        res = df["metaclass"].value_counts()
        # Reset the index of the result to convert it into a DataFrame
        res2 = res.reset_index()
        # Rename the columns of the resulting DataFrame
        res2.columns = ["metaclass", "count"]
        # Convert the categories from strings to integers using metaclass_to_int
        res2["metaclass"] = res2["metaclass"].apply(self.metaclass_to_int)
        # Convert the resulting DataFrame into a dictionary
        result = dict(zip(res2["metaclass"], res2["count"]))
        # Return the list of tuples
        return result

    def plot_class_distribution(self):
        """This method create a plot showing the distribution of images per class"""
        df = self.image_df
        res = df["metaclass"].value_counts()

        res2 = res.reset_index()
        res2.columns = ["metaclass", "count"]
        total = res2["count"].sum()

        name_map = self.get_name_map()

        # Create the bar graph
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(res2["metaclass"], res2["count"], color="skyblue")

        for bar, metaclass, count in zip(bars, res2["metaclass"], res2["count"]):
            height = bar.get_height()
            ax.annotate(
                f"{name_map[metaclass]}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
            ax.annotate(
                f"{count}",
                xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                xytext=(0, 0),  # No offset
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )

        # Agregar una anotación para la suma total
        ax.annotate(
            f"Total images: {total}",
            xy=(0.5, 1.05),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=12,
            color="black",
        )

        ax.set_title("Number of images in each category", pad=40)
        ax.set_xlabel("Metaclass")
        ax.set_ylabel("Count")
        plt.show()
