import pandas as pd
import random as rnd
from pandas import DataFrame, Series
from PIL import Image
import numpy as np
from dataclasses import dataclass
from typing import *
from pathlib import Path
from enum import Enum
import math
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import torch
import torchvision.transforms.v2 as transforms
import shutil
from tqdm import tqdm
from .cache_image_folder import CacheImgFolder


class ImgAxes(Enum):
    """Enum for denoting the type of axes in a Matrix"""

    Row = 0
    Column = 1


@dataclass
class PlotImgSerie:
    """Utility Dataclass for plotting a series of images"""

    name: str
    """Name of the serie. If set to None it will print no name"""
    images: list[Image.Image]
    """The list of images to plot"""


class BaseImagePlotter:
    """Base class for plotting images"""

    @staticmethod
    def plot_images(
        images: list[Image.Image],
        fixed_axes: tuple[ImgAxes, int] = (ImgAxes.Column, 4),
        title: str = None,
    ):
        amount = len(images)
        rows, cols = math.ceil(amount / fixed_axes[1]), fixed_axes[1]

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(3 * cols, 3 * rows))

        for row in range(rows):
            for col in range(cols):
                img = images[cols * row + col]
                ax = axes[row, col]
                ax.imshow(img)
                ax.axis("off")
        # Añadir el título en la parte inferior
        fig.suptitle(
            t=title,
            fontsize=18,
            y=0.05,
        )

        plt.tight_layout()
        plt.subplots_adjust(
            bottom=0.08,
        )  # Ajustar para que el título no se superponga
        plt.show()

    @staticmethod
    def plot_series(
        series: list[PlotImgSerie],
        fixed_axe: ImgAxes = ImgAxes.Row,
        title: str = None,
    ):
        if len(series) == 0:
            return
        max_series_length = max(len(serie.images) for serie in series)

        rows, cols = max_series_length, len(series)
        if fixed_axe == ImgAxes.Row:
            rows, cols = cols, rows

        # Create the figure and the axes
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(3 * cols, 3 * rows))

        # Plot the images
        for row, serie in enumerate(series):
            for col, image in enumerate(serie.images):
                if isinstance(image, torch.Tensor):
                    image = image.permute(1, 2, 0).numpy()
                if fixed_axe == ImgAxes.Row:
                    ax = axes[row, col]
                else:
                    ax = axes[col, row]
                ax.imshow(image)
                ax.axis("off")

        heads = axes[:, 0] if fixed_axe == ImgAxes.Row else axes[0, :]
        for ax, name in zip(heads, map(lambda s: s.name, series)):
            ax.set_title(
                name,
                fontsize=14,
            )

        # Añadir el título en la parte inferior
        fig.suptitle(
            t=title,
            fontsize=18,
            y=0.05,
        )

        plt.tight_layout()
        plt.subplots_adjust(
            bottom=0.08,
        )  # Ajustar para que el título no se superponga
        plt.show()


class ClassificationImgHelper:
    """An utility class for plotting useful information for datasets
    of images"""

    def __init__(
        self,
        dataframe: DataFrame,
        get_category: Callable[[DataFrame, str], DataFrame],
        get_image_path: Callable[[DataFrame], list[str]],
        cache: CacheImgFolder,
        save_plot_folder: Path = Path("./plots"),
    ) -> None:
        self.dataframe: DataFrame = dataframe
        """The dataframe containing path of images"""
        self.get_category: Callable[[DataFrame, str], DataFrame] = get_category
        """Gets all images in a category"""
        self.get_image_path: Callable[[DataFrame], str] = get_image_path
        """Get the image path of a row series in the dataframe"""
        self.cache: CacheImgFolder = cache
        """The cache for images"""
        self.save_plot_folder: Path = save_plot_folder
        """The path to where the plots will be saved"""

    def get_all_images_in_category(self, category: str) -> DataFrame:
        """Returns a new dataframe where all the elements belong to the same category"""
        return self.get_category(self.dataframe, category)

    def get_random_sample_with_category(
        self,
        category: str,
        amount: int,
        seed: int = None,
        replace: bool = False,
    ) -> tuple[str, list[Path]]:
        """This method returns a list of Paths of the 'amount' of images
        selected at random of the selected 'category'"""
        if seed is not None:
            rnd.seed(seed)
        df = self.get_all_images_in_category(category=category)
        amount = min(max(0, amount), len(df))
        if replace:
            rows = rnd.choices(population=range(len(df)), k=amount)
        else:
            rows = rnd.sample(population=range(len(df)), k=amount)
        df = df.iloc[rows]
        return category, list(map(Path, self.get_image_path(df)))

    def get_random_sample(
        self,
        amount: int,
        seed: int = None,
        replace: bool = False,
    ) -> list[Path]:
        """This method returns a list of Paths of the amount of images
        selected at random"""
        if seed is not None:
            rnd.seed(seed)
        amount = min(max(0, amount), len(self.dataframe))
        if replace:
            rows = rnd.choices(population=range(len(self.dataframe)), k=amount)
        else:
            rows = rnd.sample(population=range(len(self.dataframe)), k=amount)
        df = self.dataframe.iloc[rows]
        return list(map(Path, self.get_image_path(df)))

    def get_random_sample_of_categories(
        self,
        categories: list[str],
        amount: int,
        seed: int = None,
        replace: bool = False,
    ) -> list[tuple[str, list[Path]]]:
        output: list[tuple[str, list[Path]]] = []
        for category in categories:
            sample = self.get_random_sample_with_category(
                category=category,
                amount=amount,
                seed=seed,
                replace=replace,
            )
            output.append(sample)
        return output

    def plot_random_sample_of_category(
        self,
        category: str,
        amount: int,
        resolution: tuple[int, int] = (512, 512),
        dull_razor: bool = False,
        fixed_axes: tuple[ImgAxes, int] = (ImgAxes.Column, 4),
        seed: int = None,
    ):
        sample: list[Path] = self.get_random_sample_with_category(
            category=category,
            amount=amount,
            seed=seed,
        )[1]
        images = list(
            map(
                lambda p: self.cache.get_image_in_path(
                    p, resolution=resolution, dull_razor=dull_razor
                ),
                sample,
            )
        )
        rows, cols = math.ceil(amount / fixed_axes[1]), fixed_axes[1]
        if fixed_axes[0] is ImgAxes.Row:
            rows, cols = cols, rows

        BaseImagePlotter.plot_images(
            images=images,
            fixed_axes=fixed_axes,
            title=category,
        )

    def plot_random_sample(
        self,
        categories: list[str],
        amount: int,
        resolution: tuple[int, int] = (512, 512),
        dull_razor: bool = False,
        seed: int = None,
    ):
        sample: list[tuple[str, list[Path]]] = self.get_random_sample_of_categories(
            categories=categories,
            amount=amount,
            seed=seed,
        )
        get_imgs = lambda paths: list(
            map(
                lambda p: self.cache.get_image_in_path(
                    p, resolution=resolution, dull_razor=dull_razor
                ),
                paths,
            )
        )
        series = [PlotImgSerie(cat, get_imgs(images)) for cat, images in sample]
        BaseImagePlotter.plot_series(
            series=series, fixed_axe=ImgAxes.Row, title="Sample of images per category"
        )

    def plot_random_sample_with_filters(
        self,
        amount: int,
        filters: list[tuple[Callable[[Image.Image], Image.Image], str]],
        resolution: tuple[int, int] = (512, 512),
        dull_razor: bool = False,
        seed: int = None,
    ):
        sample = self.get_random_sample(amount=amount, seed=seed)
        images = list(
            map(
                lambda p: self.cache.get_image_in_path(
                    p,
                    resolution=resolution,
                    dull_razor=dull_razor,
                ),
                sample,
            )
        )
        to_plot_img_serie = lambda name, imgs: PlotImgSerie(name=name, images=imgs)
        apply_filter = lambda filter_name, filter_fn, imgs: (
            filter_name,
            list(filter_fn(img) for img in imgs),
        )

        originals = to_plot_img_serie("Originals", images)
        filtered_imgs = [
            apply_filter(filter_name, filter_fn, images)
            for filter_fn, filter_name in filters
        ]
        filtered_series = [
            to_plot_img_serie(name=f[0], imgs=f[1]) for f in filtered_imgs
        ]

        filtered_series.insert(0, originals)
        BaseImagePlotter.plot_series(
            filtered_series,
            fixed_axe=ImgAxes.Row,
            title="Example of images with filters",
        )


class SegmentationImgHelper:
    def __init__(
        self,
        dataframe: DataFrame,
        get_images_in_df: Callable[[DataFrame], list[Path]],
        get_masks_in_df: Callable[[DataFrame], list[Path]],
        cache: CacheImgFolder = CacheImgFolder("./cache/segmentation"),
    ) -> None:
        self.dataframe: DataFrame = dataframe
        """The dataframe containing the information about the segmentation dataset"""
        self.cache: CacheImgFolder = cache
        """The cache for helping in obtaining the images"""
        self.get_images_in_df: Callable[[DataFrame], list[Path]] = get_images_in_df
        """A function to extract a list with the paths of all of the images in the dataframe"""
        self.get_masks_in_df: Callable[[DataFrame], list[Path]] = get_masks_in_df
        """A function to extract a list with the paths of all of the masks in the dataframe"""

    def get_random_sample(
        self,
        amount: int = 4,
        replace: bool = False,
        seed: int = None,
    ) -> tuple[list[Path], list[Path]]:
        """This method extracts a tuple `(images, masks)` with each pair been
        a list of paths to images and masks whose `length` is `amount`"""
        if seed is not None:
            rnd.seed(seed)
        amount = min(max(0, amount), len(self.dataframe))
        if replace:
            rows = rnd.choices(population=range(len(self.dataframe)), k=amount)
        else:
            rows = rnd.sample(population=range(len(self.dataframe)), k=amount)
        df = self.dataframe.iloc[rows]
        return (self.get_images_in_df(df), self.get_masks_in_df(df))

    @staticmethod
    def apply_colored_mask(
        image: Image.Image,
        mask: Image.Image,
        color: tuple[float, float, float] = (0, 0, 1),
        alpha: float = 0.2,
    ):
        # Convert images from PIL to normalized numpy array
        image_np = np.array(image).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.float32) / 255.0

        # Create a mask of the size of the original image
        colored_mask = np.zeros_like(image_np)
        for i in range(3):
            colored_mask[:, :, i] = mask_np * color[i]

        overlay = image_np * (1 - alpha) + colored_mask * alpha
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
        return overlay_img

    @staticmethod
    def apply_colored_mask_2(
        image: Image.Image,
        mask: Image.Image,
        color_map: str = "inferno",
        alpha: float = 0.2,
    ):
        image_np = np.array(image.convert("RGB"))
        mask_np = np.array(mask.convert("L"))

        normalized_mask = mask_np / 255.0

        cmap = get_cmap(name=color_map).copy()
        cmap.set_under("black")
        norm = Normalize(vmin=0.1, vmax=1)

        colored_mask = cmap(norm(normalized_mask))
        colored_mask_rgb = (colored_mask[:, :, :3] * 255).astype(np.uint8)

        black_pixels_colored_mask = (colored_mask_rgb == [0, 0, 0]).all(axis=-1)
        colored_mask_rgb[black_pixels_colored_mask] = image_np[
            black_pixels_colored_mask
        ]

        result = (image_np * (1 - alpha) + colored_mask_rgb * alpha).astype(np.uint8)
        return Image.fromarray(result)

    def plot_imgs_with_masks(
        self,
        amount: int,
        resolution: tuple[int, int] = (512, 512),
        color_map: str = "inferno",
        overlay_alpha: float = 0.3,
        dull_razor: bool = False,
        seed: int = None,
    ):
        sample: tuple[list[Path], list[Path]] = self.get_random_sample(
            amount=amount, seed=seed
        )
        get_images = lambda paths: [
            self.cache.get_image_in_path(
                relative_image_path=p,
                resolution=resolution,
                dull_razor=dull_razor,
            )
            for p in paths
        ]

        images, masks = get_images(sample[0]), get_images(sample[1])

        overlay_image = [
            self.apply_colored_mask_2(
                image=image,
                mask=mask,
                alpha=overlay_alpha,
                color_map=color_map,
            )
            for image, mask in zip(images, masks)
        ]

        img_serie, mask_serie = PlotImgSerie(name="Image", images=images), PlotImgSerie(
            name="Mask", images=masks
        )
        overlay_image_serie = PlotImgSerie(name="Image + Mask", images=overlay_image)

        BaseImagePlotter.plot_series(
            [img_serie, mask_serie, overlay_image_serie],
            fixed_axe=ImgAxes.Column,
            title="Images with corresponding masks",
        )
