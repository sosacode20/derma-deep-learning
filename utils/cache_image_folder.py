from pathlib import Path
from PIL import Image
from pandas import DataFrame
import shutil
from mpire import WorkerPool
from tqdm import tqdm
from typing import *

class CacheImgFolder:
    """Utility class for implementing a cache system for images"""

    def __init__(
        self,
        root_image_folder: Path,
        cache_image_folder: Path = Path("./cache"),
        default_filter: Callable[[Image.Image], Image.Image] = None,
    ) -> None:
        self.root_image_folder: Path = root_image_folder
        """The base path for searching images if not found in the cache"""
        self.cache_image_folder: Path = cache_image_folder
        """The base path pointing to where te cache will store it's data"""
        self.default_resolution: tuple[int, int] = (512, 512)
        """The default resolution for the images"""
        self.enabled: bool = True
        """If enabled then the images are searched first in cache.
        If not enabled it will always search from root directory"""
        self.default_filter: Callable[[Image.Image], Image.Image] = default_filter
        """The default filter to apply to the images"""

    def disable_cache(self):
        """It disable the cache"""
        self.enabled = False

    def enable_cache(self):
        """It enable the cache"""
        self.enabled = True

    def _setup_cache(self):
        """Setups the cache folder"""
        self.cache_image_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def open_image(path: Path):
        """Opens an image from a path"""
        try:
            image = Image.open(path)
            image.verify()
            image = Image.open(path)
            return image
        except Exception as e:
            raise Exception(f"Error opening image '{path}': {e}")

    def get_image_in_path(
        self,
        relative_image_path: Path,
        resolution: tuple[int, int] = None,
        use_filter: bool = False,
    ) -> Image.Image:
        """Extract the image"""
        self._setup_cache()
        if resolution is None:
            resolution = self.default_resolution
        cache_image_path = (
            self.cache_image_folder
            / f"{resolution[0]}X{resolution[1]}"
            / relative_image_path
        )
        found_in_cache: bool = False
        if cache_image_path.exists() and self.enabled:
            image = self.open_image(cache_image_path)
            found_in_cache = True
        else:
            original_path = self.root_image_folder / relative_image_path
            image = self.open_image(original_path)
            image = image.resize(size=resolution)

        if self.enabled and not found_in_cache:
            cache_image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(cache_image_path)
        if use_filter and self.default_filter is not None:
            image = self.default_filter(image=image)
        return image

    def remove_cache_content(self):
        """Removes all the content of the cache"""
        shutil.rmtree(self.cache_image_folder)

    def preprocess_folder_in_cache(
        self,
        dataframe: DataFrame,
        path_column_name: str,
        resolution: tuple[int, int] = None,
    ):
        """Preprocess all the images in the folder"""

        self._setup_cache()
        cache_status = self.enabled
        self.enable_cache()

        process_image = lambda path: self.get_image_in_path(
            relative_image_path=Path(path),
            resolution=resolution,
        )

        image_paths = dataframe[path_column_name].tolist()

        # Usar WorkerPool de mpire para ejecutar process_image en paralelo
        with WorkerPool(
            n_jobs=4
        ) as pool:  # Puedes ajustar n_jobs según el número de núcleos de tu CPU
            for _ in tqdm(
                pool.imap_unordered(process_image, image_paths),
                total=len(image_paths),
                colour="green",
            ):
                pass
        self.enabled = cache_status
