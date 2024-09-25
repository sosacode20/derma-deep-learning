from pathlib import *
from typing import *
from PIL import Image

def get_all_files_with_extensions(
    root_directory: Path, extensions: list[str], relative: bool = True
) -> Generator[Path, Any, None]:
    """Returns all the files in the root directory and subfolders
    that match some of the extensions listed in `extensions`"""
    for path in root_directory.rglob("*"):
        if path.is_dir() or path.is_symlink() or path.suffix[1:] not in extensions:
            continue
        if relative:
            yield path.relative_to(root_directory)
        else:
            yield path

def is_valid_image(image_path: Path) -> bool:
    """Returns True if the image file is valid, False otherwise"""
    try:
        if image_path.stat().st_size == 0:
            return False
        with Image.open(image_path) as img:
            img.verify()  # Verifica la integridad de la imagen
            return True
    except (IOError, ValueError):
        return False
    return False


def get_all_images_with_parent_folder_name(
    root_directory: Path, relative: bool = True
) -> Generator[tuple[str, Path], Any, None]:
    """Returns a generator of tuples (parent_folder_name:str, image_path:Path)
    where the `parent_folder_name` is a string representing the name of the parent folder where the image was found.
    This is useful for the names of skin lesions.
    The `image_path` is an object of type Path that points to where the image was found.
    If you set the parameter `relative` to `True` you will have the path of the image relative to the `root_directory` path.
    """
    for image in get_all_files_with_extensions(
        root_directory, extensions=["png", "jpg", "jpeg"], relative=relative
    ):
        if is_valid_image(root_directory / image):
            yield image.parent.name, image
