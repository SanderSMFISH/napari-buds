import numpy as np
from skimage.data import astronaut

from .._widget import Main

def test_Main(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    img = viewer.add_image(astronaut()[:,:,0],name='Image')

    labels=np.zeros(img.data.shape,dtype=int)
    label=1
    for i in range(0,100,34):
        labels[:,i]=label
        label+=1
    Labels= viewer.add_labels(labels,name='Labels')

    # create our widget, passing in the viewer
    my_widget = Main(viewer)
    my_widget.train.train_classify()
    my_widget.train.classify()
    my_widget.maxima(image=img, threshold=99)
    my_widget.threshold(image=img, threshold=99)
    my_widget.segment()
    my_widget.draw_mother_bud()

    #refresh GUI functions
    my_widget.update_layer_extraction_container()