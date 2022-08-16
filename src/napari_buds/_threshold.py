from magicgui import magic_factory
from scipy.ndimage import label as nlabel 
from skimage.morphology import binary_erosion, binary_dilation
from magicgui import magic_factory
from napari.types import ImageData, LabelsData

def erosion(img, iter):
    for i in range(0,iter):
        img=binary_erosion(img)
    return img

def dilation(img, iter):
    for i in range(0,iter):
        img=binary_dilation(img)
    return img

@magic_factory(auto_call=True,call_button='Threshold')
def threshold(image: ImageData, threshold: int = 100) -> LabelsData:
    image =(image*(100/image.max()))
    seeds=image>threshold
    seeds=erosion(seeds,3)
    seeds=dilation(seeds,3)
    markers,_=nlabel(seeds)

    return markers

@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")