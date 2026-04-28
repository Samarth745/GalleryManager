#@title image_utils.py
import os
import requests
from PIL import Image
from io import BytesIO
from PIL import Image
import random
import math
import config.config_app as config

GS_KEY = "source"

def load_image(image_source: str) -> Image.Image:
    """
    Loads an image from a local file path or a URL.

    Args:
        image_source (str): Path to a local image or a valid URL (http/https).

    Returns:
        PIL.Image.Image: Loaded image object.

    Raises:
        FileNotFoundError: If local file is not found.
        requests.RequestException: If the image URL is invalid or unreachable.
        PIL.UnidentifiedImageError: If the image cannot be opened.
    """
    if image_source.startswith("http://") or image_source.startswith("https://"):
        try:
            response = requests.get(image_source, timeout=10, verify=False)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to load image from URL: {e}")
    else:
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"Image not found at path: {image_source}")
        return Image.open(image_source)


def get_all_images(root_dir=config.ORIGINAL_IMAGES_DIR, exts=('jpg', 'jpeg', 'png')):
    """
    Recursively get all image file paths under root_dir with given extensions.
    Returns a list of full paths.
    """
    images = []
    for dirpath, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(exts):
                images.append(os.path.join(dirpath, file))
    return images


def show_image_grid(images, limit=10, thumb_size=(128, 128), background_color=(255, 255, 255)):
    if limit is None:
        limit = len(images)
    images = random.sample(images, min(limit, len(images)))
    n = len(images)
    
    # Calculate rows and columns for the grid
    rows = int(math.sqrt(n))
    cols = (n + rows - 1) // rows  # Ceiling division to fit all images

    # Create empty white background image
    grid_width = cols * thumb_size[0]
    grid_height = rows * thumb_size[1]
    grid_img = Image.new('RGB', (grid_width, grid_height), background_color)

    for idx, img in enumerate(images):
        img_copy = img.copy()
        img_copy.thumbnail(thumb_size)

        # Calculate position
        row = idx // cols
        col = idx % cols
        x = col * thumb_size[0]
        y = row * thumb_size[1]

        # Paste thumbnail into grid
        grid_img.paste(img_copy, (x, y))

    return grid_img

