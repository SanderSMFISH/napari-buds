from magicgui import magicgui,magic_factory
from magicclass import HasFields, vfield, magicclass, set_design
from magicgui.widgets import LineEdit, SpinBox, Container, PushButton, create_widget
from skimage import io,segmentation,data, feature, future
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt as distance
from skimage import future
from napari.types import LabelsData, ImageData
from functools import partial
import napari

def create_fs(channels,clf,viewer:napari.Viewer,training=False):
        fs=[]
        for x in viewer.layers:
            if x.name in channels or x.name=='DIC_new':
                array=viewer.layers[x.name].data.astype(np.uint16)
                fs.append(features_func(array))
        features = np.concatenate(fs, axis=-1)
        if training==False:
            result = future.predict_segmenter(features, clf)
        else:
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                             max_depth=10, max_samples=0.05)
            training_labels = viewer.layers['Labels'].data.astype(np.uint16)
            clf = future.fit_segmenter(training_labels, features, clf)
            named_tuple = time.localtime()
            time_string = time.strftime("%m%d%Y", named_tuple)
            classifier_name=f'class_{time_string}'
            #dump(clf, classifier_name)
            result = future.predict_segmenter(features, clf)
        viewer.add_labels(result,opacity=0.5)

        return result,clf,classifier_name

@magicclass
class Train_Classifier():
    """
    Training and Apply new classifiers. Additionally define the Class Labels
    """
    def __init__(self):
        self.class_labels =  ['cell','bud','background']
        self.features_func = None
        self.clf = None
        self.counter = 0.0
        self.result = None
        self.dict_labels= {}

    @set_design(text="Set random forest parameters")
    def set_random_forest_params(self,intensity_bool=True, edges_bool=True, texture_bool=True,
                        sigma_min=1, sigma_max=20):
        self.features_func=partial(feature.multiscale_basic_features,
                        intensity=intensity_bool, edges=edges_bool, texture=texture_bool,
                        sigma_min=sigma_min, sigma_max=sigma_max)

    @set_design(text="Train classifier")
    def train_classify(self): 
        clf=None
        self.clf,self.result=create_fs(channels,clf,training=False)

    @set_design(text="Classify")
    def classify(self): 
            self.clf,self.result=create_fs(channels,clf,training=True)
   
    @viewer.layers.changed.connect
    def def_class_labels(self,viewer:napari.Viewer):
        labels=np.unique(viewer.layers['Labels'].data)
        labels=list(np.delete(labels, np.where(labels == 0)))
        labels=[f'class_{label}' for label in labels]
        slabels=['cell','bud','background']

        if len(labels)>len(slabels):
            i=len(labels)
            j=len(slabels)
            slabels = slabels + labels[j:i]
            
        container = Container(widgets=[create_widget(label, name=slabel) for slabel,label in zip(labels,slabels)])
        viewer.window.add_dock_widget(container, name='Define_class_labels')
        
        @container.changed.connect
        def save_new_class_labels():
            self.dict_labels=container.asdict()
            print(self.dict_labels)