import numpy as np
from .._widget_overview import UIWidget

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
    my_widget = UIWidget(viewer)
    my_widget.Train_Classifier_widget.train_classifier()
    my_widget.Train_Classifier_widget.classify()
    #my_widget.Maxima.create_seeds()
    my_widget.Threshold.create_seeds()
    my_widget.Segment.segment_cells()
    
    viewer.layers.remove('cell mask')
    viewer.layers.remove('buds')

    viewer.add_labels(np.array([[1, 2], [3, 4]]),name='cell mask')
    viewer.add_labels(np.array([[1, 2], [3, 4]]),name='buds')

    my_widget.Draw.draw_mother_bud()