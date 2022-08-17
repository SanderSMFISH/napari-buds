import numpy as np

from .._widget import Main

def test_Main(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))
    layer = viewer.add_labels(np.zeros((100, 100),dtype=int))

    # create our widget, passing in the viewer
    my_widget = Main(viewer)

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == f"you have selected {layer}\n"
