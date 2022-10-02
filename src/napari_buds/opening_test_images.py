from glob import glob 
from skimage import io
import napari
import numpy as np

path_imgs=r'D:\Master\Internship Filamentous Yeast\segmentation_code\ASH1-CLB2 bud marker\YET862\220222\TIFs\*.tif'
#select path to DICs
path_DICs=r'D:\Master\Internship Filamentous Yeast\segmentation_code\ASH1-CLB2 bud marker\YET862\220222\DIC\*.tif'
#mask for testing
path_mask=r'D:\Master\Internship Filamentous Yeast\segmentation_code\ASH1-CLB2 bud marker\YET862\220222\*.tif'
#mask for seeds
path_seeds=r'D:\masks\seeds.tif'
path_results=r'D:\masks\result.tif'

#select fov to observe; which image in folder
fov=0
#shape of your images (c,z,x,y)
shape=(4,41,2304,2304)
#which layers are in focus?
focus=[20,30]
#channel order
channels=['CY3_5','CY3','DAPI','CY5']
#colors to display different channels in Napari
colors=['green','red','blue','bop purple']
#set intensity ranges (these can be adjusted in napari as well).
cls=[[4530,5908],[4664,5110],[11658,30000],[7462,10156]]
#class labels
class_labels=['cell','bud','background']

#read files
filenames=glob(path_imgs)
img=io.imread(filenames[fov], plugin='pil')
print(img.shape)
img=np.reshape(img,shape)
DIC_filenames=glob(path_DICs)
DIC=io.imread(DIC_filenames[fov])
print(img.shape,DIC.shape)

#zproject
zproject=np.amax(img[:,focus[0]:focus[1],...],axis=1,keepdims=False)
zproject.shape

#mask for testing
mask=io.imread(path_mask)

viewer= napari.Viewer()
viewer.add_image(DIC, name='DIC', opacity=0.50) #contrast_limits=[5760,13046])
for i in range(zproject.shape[0]):
    viewer.add_image(zproject[i,...],name=channels[i],colormap=colors[i],opacity=0.60,blending='additive')   #contrast_limits=cls[i])
viewer.add_labels(mask,name='Labels')

#viewer.add_labels(io.imread(path_seeds),name='seeds')
#viewer.add_labels(io.imread(path_results),name='result')
napari.run()
#python ./src/napari_buds/opening_test_images.py 