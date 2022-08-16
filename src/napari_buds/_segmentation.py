from _segmentation_functions import erosion,dilation, watershed_seg, draw_mother_bud_relations
from magicgui import magicgui, magic_factory
from napari.layers import Labels, Image
from napari.types import LabelsData
import napari
from scipy.ndimage import label as nlabel

#threshold seeds
@magic_factory(auto_call=False,call_button='Threshold')
def threshold(image: Image, threshold: int = 100)-> LabelsData:
    image =(image.data*(100/image.data.max()))
    seeds=image>threshold
    seeds=erosion(seeds,3)
    seeds=dilation(seeds,3)
    markers,_=nlabel(seeds)

    print(markers.shape)
    return markers

@magic_factory(auto_call=False,call_button='Segment')
def segment()-> LabelsData:
    viewer=napari.viewer.Viewer
    result=viewer.layers['result'].data
    label, labeled_buds=watershed_seg(result)

    return label
    #draw_mother_bud_relations(label, labeled_buds)