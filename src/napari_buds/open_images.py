import napari 
import numpy as np
from skimage.data import coins

coins=coins()
coins=np.array(coins,dtype=int)

viewer=napari.Viewer()
viewer.add_image(coins, name='coins')

mask=np.zeros(viewer.layers['coins'].data.shape,dtype=int)
viewer.add_labels(mask,name='Labels')

napari.run()