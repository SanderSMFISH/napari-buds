import numpy as np

from .._widget import Main

def test_Main(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    img = viewer.add_image(np.array([[1, 2,3], [3, 4,5]]),name='Image')

    labels=np.zeros(img.data.shape,dtype=int)
    label=1
    for i in range(1,4):
        labels[:,i-1]=label
        label+=1
    Labels= viewer.add_labels(labels,name='Labels')

    # create our widget, passing in the viewer
    my_widget = Main(viewer)
    my_widget.train.train_classify()
    my_widget.train.classify()
    my_widget.maxima(image=img, threshold=10)
    my_widget.threshold(image=img, threshold=10)
    my_widget.segment()
    
    viewer.layers.remove('cell mask')
    viewer.layers.remove('buds')

    viewer.add_labels(np.array([[1, 2], [3, 4]]),name='cell mask')
    viewer.add_labels(np.array([[1, 2], [3, 4]]),name='buds')

    my_widget.draw_mother_bud()