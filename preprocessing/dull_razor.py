import cv2
from PIL import Image
import numpy as np
import random as rnd
from torchvision.transforms.v2 import Transform, ColorJitter
import torch
import math


def dull_razor(image: Image, kernel_size: tuple[int, int]) -> Image:
    # Convert PIL image to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Gray scale
    grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Black hat filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # Gaussian filter
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
    # Binary thresholding (MASK)
    ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    # Replace pixels of the mask
    dst = cv2.inpaint(image, mask, 6, cv2.INPAINT_TELEA)

    # Convert the resulting image to PIL format
    pil_image_result = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    return pil_image_result


class DullRazor(torch.nn.Module):

    def __init__(
        self,
        kernel_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.kernel_size: tuple[int, int] = kernel_size
        """The kernel size for the filter"""

    def forward(self, *args):
        new_args = tuple(
            (
                dull_razor(arg, kernel_size=self.kernel_size)
                if isinstance(arg, Image.Image)
                else arg
            )
            for arg in args
        )

        return new_args[0] if len(new_args) == 1 else new_args


class RandomDullRazor(torch.nn.Module):
    """A transformation over PIL images that removes almost all hair from them"""

    def __init__(
        self,
        min_kernel_size: int,
        max_kernel_size: int,
        probability: float,
    ):
        super().__init__()
        self.min_kernel_size: int = max(1, min_kernel_size)
        """The minimum allowable kernel size"""
        self.max_kernel_size: int = max(min_kernel_size, max_kernel_size)
        """The maximum allowable kernel size"""
        self.probability: float = probability
        """The probability of applying the transform"""

    @staticmethod
    def from_image_size(image_size: tuple[int, int], probability:float = 1) -> "RandomDullRazor":
        """Creates a DullRazor instance from the image size wth appropriate kernel size"""
        percentage = lambda total, percentage: math.floor((total * percentage) / 100)
        probability = max(0, min(1, probability))

        total = min(image_size)
        minimum, maximum = percentage(total, 1), percentage(total, 1.5)
        minimum, maximum = max(minimum, 3), max(maximum, 3)
        return RandomDullRazor(minimum, maximum, probability=probability)

    def forward(self, *args):
        if rnd.random() < self.probability:
            kernel_size = rnd.randint(self.min_kernel_size, self.max_kernel_size)
            args = tuple(
                (
                    dull_razor(arg, kernel_size=(kernel_size, kernel_size))
                    if isinstance(arg, Image.Image)
                    else arg
                )
                for arg in args
            )
        return args[0] if len(args) == 1 else args
