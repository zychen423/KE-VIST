import sys
from PIL import Image

img_path = sys.argv[1]
out_path = sys.argv[2]

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

with open(img_path, 'r+b') as f:
    try:
        with Image.open(f) as img:
            img = resize_image(img, [256,256])#224
            img.save(out_path, img.format)
    except:
        print(img_path,'badbad')
