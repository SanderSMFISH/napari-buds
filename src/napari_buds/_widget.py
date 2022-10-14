from magicgui import magicgui,magic_factory
from magicclass import HasFields, vfield, magicclass, set_design, set_options
from magicclass.types import OneOf
from magicgui.widgets import LineEdit, SpinBox, Container, PushButton, create_widget, Label
from skimage import io,segmentation,data, feature, future
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import numpy as np
import pickle
import pandas as pd
import joblib
from scipy.ndimage import label as nlabel
from scipy.ndimage import distance_transform_edt as distance
from skimage import future
from napari.types import LabelsData, ImageData
from functools import partial
import napari
from typing import TYPE_CHECKING
import time
from napari import Viewer
import numpy as np
from magicgui.widgets import Container,create_widget
from qtpy.QtWidgets import (QWidget,
                            QVBoxLayout,
                            QTabWidget,
                            QPushButton)
from superqt import QCollapsible
from magicgui import widgets
from napari.qt.threading import thread_worker
from functools import partial
from napari.layers import Image, Labels
from ._segmentation_functions import dilation,erosion,watershed_seg, clean_up, label_id, draw_mother_bud_relations, count_class_labels
from qtpy.QtWidgets import QWidget, QMainWindow, QApplication, QDockWidget,QScrollArea
from qtpy.QtCore import QObject, QEvent, Qt
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu

from typing import TYPE_CHECKING


from magicgui import magic_factory, widgets
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QCheckBox

if TYPE_CHECKING:
    import napari

#main napari widget for bud-annotation plugin
class Main(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.outer_scroll = QScrollArea() 
        self.vbox = QVBoxLayout()
        self.viewer = napari_viewer
        self.installEventFilter(self)
        self.class_labels =  ['cells','buds','background']
        self.savefolder=Path.home()
        self.clf = None
        self.hidden_layers=['result','seeds','Labels','relations mother buds','cell mask','distance']
        
        #check whether label layer is added to the viewer.
        try:
            self.viewer.layers['Labels']
        except KeyError:
            print(f"create a label layer first!")
            
        #extract layer names for feature extraction
        labels_FE=[self.viewer.layers[i].name for i in range(len(self.viewer.layers))][::-1]
        labels_FE=[x for x in labels_FE if x not in self.hidden_layers]
        layers_to_select = Container(widgets=[create_widget(name=label, widget_type='CheckBox',value=1) for label in labels_FE])
        #layers_to_select.insert(0,Label(name='Layers_to_extract_Features_from:'))
        self.labels_FE=labels_FE

        #update layer choices after changes to label layer
        @self.viewer.layers.events.connect
        def update_layer_extraction_container():
            labels_FE=[self.viewer.layers[i].name for i in range(len(self.viewer.layers))][::-1]
            labels_FE=[x for x in labels_FE if x not in self.hidden_layers]
            old_layers=layers_to_select.asdict()
            #old_layers.pop('Layers_to_extract_Features_from:')
            for layer in old_layers.keys():
                if layer not in labels_FE:
                    layers_to_select.remove(layer)
            i=1
            for layer in labels_FE:
                if layer not in old_layers:
                    layers_to_select.insert(i,create_widget(name=layer, widget_type='CheckBox'))
                i+=1
            self.labels_FE=labels_FE

        #define class labels
        slabels=self.class_labels
        widget_names = count_class_labels(self.viewer.layers['Labels'].data)

        if len(widget_names)>len(slabels) & len(widget_names)!=0:
            i=len(widget_names)
            j=len(slabels)
            slabels = slabels + widget_names[j:i]

        if len(widget_names)<len(slabels):
            i=len(widget_names)
            j=len(slabels)
            slabels = slabels[:i]
        self.class_labels=slabels

        #add class labels define widget to GUI
        labels_to_define = Container(widgets=[create_widget(slabel,name=widget_name) for widget_name,slabel in zip(widget_names,slabels)])
        labels_to_define_tag=Container(widgets=[Label(name='Define_Label_names:')],labels=True)
        Refresh_labels=PushButton(name="Refresh")
        
        #PushButton to refresh class labels that are defined in the GUI
        Refresh_labels.is_connected = False
        @Refresh_labels.changed.connect
        def _on_connect(event):
            Refresh_labels.is_connected = not Refresh_labels.is_connected
            Refresh_labels.text = 'Refreshed' if Refresh_labels.is_connected else 'Refreshed'

            #redefine class labels
            slabels =  self.class_labels
            pot_widget_names = count_class_labels(self.viewer.layers['Labels'].data)
            
            if len(pot_widget_names)!=len(slabels):
                i=len(pot_widget_names)
                j=len(slabels)
                slabels = slabels + pot_widget_names[j:i]

                for widget_name,slabel, in labels_to_define.asdict().items():
                    if widget_name not in pot_widget_names:
                        labels_to_define.remove(widget_name)
                for pot_widget_name in pot_widget_names:
                    if pot_widget_name not in labels_to_define.asdict().keys():
                        labels_to_define.append(create_widget(pot_widget_name,name=pot_widget_name))

            self.class_labels=list(labels_to_define.asdict().values())

            self.label = self.viewer.layers['cell mask'].data
            self.labeled_buds = self.viewer.layers['buds'].data
            


        #magicclass automatic widget creation for Random forest classification, training and saving + loading of classifiers.
        @magicclass(layout='vertical', widget_type = "collapsible", name = "Random forest classification")
        class Train_Classifier():
            def __init__(self):
                self.viewer = napari_viewer
                self.features_func=partial(feature.multiscale_basic_features,intensity=True, edges=True, texture=True,
                                sigma_min=1, sigma_max=20)
                self.rf_params=RandomForestClassifier(n_estimators=100, n_jobs=-1,
                             max_depth=10, max_samples=0.05)

            #random forest classification parameter widget
            @set_design(text="Set random forest parameters")
            def set_random_forest_params(self,intensity=True, edges=True, texture=True,
                                sigma_min=1, sigma_max=20, n_estimators=100,n_jobs=-1,max_depth=10,max_samples=0.05):
                self.features_func=partial(feature.multiscale_basic_features,
                                intensity=intensity, edges=edges, texture=texture,
                                sigma_min=sigma_min, sigma_max=sigma_max)
                self.rf_params=RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs,
                             max_depth=max_depth, max_samples=max_samples)

            #train classifier by extracting from checked feature layers and fitting + predicting random forest parameters
            @set_design(text="Train classifier")
            def train_classify(self):
                fs_features=layers_to_select.asdict()
                print(fs_features)
                fs=[]
                for fs_feature,check in fs_features.items():
                    if check==True:
                        array=self.viewer.layers[str(fs_feature)].data.astype(np.uint16)
                        fs.append(self.features_func(array))
                features = np.concatenate(fs, axis=-1)
                print(features.shape)
                clf = self.rf_params
                training_labels = self.viewer.layers['Labels'].data.astype(np.uint32)
                clf = future.fit_segmenter(training_labels, features, clf)
                self.clf=clf
                result = future.predict_segmenter(features, clf)
                try:
                    self.viewer.remove('result')
                except:
                    pass
                self.viewer.add_labels(result,name='result',opacity=0.5)

            #classify using loaded classifier
            @set_design(text="Classify")
            def classify(self):
                fs_features=layers_to_select.asdict()
                print(fs_features)
                fs=[]
                for fs_feature,check in fs_features.items():
                    if check==True:
                        array=self.viewer.layers[str(fs_feature)].data.astype(np.uint16)
                        fs.append(self.features_func(array))
                features = np.concatenate(fs, axis=-1)
                print(features.shape)
                clf = self.clf
                training_labels = self.viewer.layers['Labels'].data.astype(np.uint32)
                result = future.predict_segmenter(features, clf)
                try:
                    self.viewer.remove('result')
                except:
                    pass
                self.viewer.add_labels(result,name='result',opacity=0.5) 
            
            #save current classifier
            @set_options(file={'mode': 'w'}, call_button='save')
            def Save_classifier(self,file = self.savefolder):
                self.savefolder=file
                joblib.dump(self.clf, str(file)) 
            
            #load a previously saved classifier            
            @set_options(file={'mode': 'r'}, call_button='Load')
            def Load_classifier(self,file=self.savefolder):
                self.savefolder=file
                self.clf=joblib.load(str(file))

        #define seeds for watershed segmentation by thresholding
        @magic_factory(auto_call=False,call_button='Threshold')
        def threshold(image: Image, threshold: int = 100):
            self.image=image
            image =(image.data*(100/image.data.max()))
            seeds=image>threshold
            seeds=clean_up(seeds)
            markers,_=nlabel(seeds)
            try:
                self.viewer.layers.remove('seeds')
            except:
                pass
            self.viewer.add_labels(markers,name='seeds')

        threshold=threshold()

        self.threshold=threshold  

        @self.viewer.layers.events.connect
        def reset_threshold_choices():
            self.threshold.reset_choices()


        #define seeds by peak local maxima on distance transformed image
        @magic_factory(threshold = {'widget_type': 'Slider','min':-100,'max':100},call_button='Find_local_maxima')
        def find_local_maxima(image:Image,threshold: int =10,min_distance: int = 20, threshold_abs: int = 15,threshold_rel: int = 0):

            if threshold_abs==0:
                threshold_abs=None
            if threshold_rel==0:
                threshold_rel=None

            thresholded_image=image.data>(threshold_otsu(image.data)+threshold*10)
            distance_image=distance(thresholded_image)
            local_max_coords = feature.peak_local_max(
            distance_image,
            min_distance=min_distance,
            threshold_rel=threshold_rel,
            threshold_abs=threshold_abs,
            )
            local_max_mask = np.zeros(distance_image.shape, dtype=bool)
            print(local_max_mask.shape)
            local_max_mask[tuple(local_max_coords.T)] = True
            local_max_mask=dilation(local_max_mask,3)
            markers, _ = nlabel(local_max_mask)
            print(markers)
            try:
                self.viewer.layers.remove('seeds')
                self.viewer.layers.remove('distance')
            except:
                pass
            self.viewer.add_image(distance_image,name='distance',opacity=0.3)
            self.viewer.add_labels(markers,name='seeds')

        maxima=find_local_maxima()
        self.maxima=maxima

        @self.viewer.layers.events.connect
        def reset_find_local_maxima():
            self.maxima.reset_choices()

        #Watershed segmentation of cell and bud layer
        @magic_factory(auto_call=False,call_button='Segment', labels=False)
        def segment():
            cell_id,bud_id,bg_id=label_id(labels_to_define.asdict())
            result=self.viewer.layers['result'].data
            seeds=self.viewer.layers['seeds'].data
            label, labeled_buds=watershed_seg(result,seeds,cell_id,bud_id,bg_id)
            self.label=label
            self.labeled_buds=labeled_buds
            try:
                self.viewer.layers.remove('cell mask')
                self.viewer.layers.remove('buds')
            except:
                pass
            self.viewer.add_labels(self.label,name='cell mask')
            self.viewer.add_labels(self.labeled_buds,name='buds')


        #draw mother bud relations
        @magic_factory(auto_call=False,call_button='Draw Mother-Bud relations', labels=False)
        def draw_mother_bud():
            label, labeled_buds, vector_image  = draw_mother_bud_relations(self.viewer.layers['cell mask'].data , self.viewer.layers['buds'].data)
            try:
                self.viewer.layers.remove('buds')
                self.viewer.layers.remove('cell mask')
                self.viewer.layers.remove('relations mother buds')
            except:
                pass
            #self.viewer.add_image(vector_plot, name='relations mother buds')
            self.viewer.add_labels(label, name='cell mask')
            self.viewer.add_labels(labeled_buds, name='buds')
            self.viewer.add_image(vector_image,name='relations mother buds',opacity=0.3)

        #organising the final gui
        #Classifier widget 
        Train_Classifier=Train_Classifier()
        self.train=Train_Classifier
        cont_Train_Classifier=Container(widgets=[Train_Classifier],labels=False)

        #seeds
        Seeds_1=Container(widgets=[self.threshold],labels=False)
        Seeds_2=Container(widgets=[self.maxima],labels=False)

        #segment
        Segment=segment()
        self.segment=Segment
        draw_mother_bud=draw_mother_bud()
        self.draw_mother_bud=draw_mother_bud
        segment_cont=Container(widgets=[Segment,draw_mother_bud],labels=False,name='Watershed_segmentation')

        #make selection scrollable to prevent overcrowding widget
        label_tag_main=Container(widgets=[Label(name='Layers_to_extract_Features_from:')],labels=True)
        self.vbox.addWidget(label_tag_main.native)
        self.scroll = QScrollArea()
        self.scroll.setWidget(layers_to_select.native)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.vbox.addWidget(self.scroll)
        self.vbox.addWidget(labels_to_define_tag.native)
        self.vbox.addWidget(labels_to_define.native)
        self.vbox.addWidget(Refresh_labels.native) 
        self.vbox.addWidget(cont_Train_Classifier.native)
        seed_tag_main=Container(widgets=[Label(name='Define_watershed_seeds')],labels=True)
        self.vbox.addWidget(seed_tag_main.native)
        self._collapse1 = QCollapsible('Thresholding', self)
        self._collapse1.addWidget(Seeds_1.native)
        self.vbox.addWidget(self._collapse1)
        self._collapse2 = QCollapsible('Distance transform:', self)
        self._collapse2.addWidget(Seeds_2.native)
        self.vbox.addWidget(self._collapse2)
        segment_tag=Container(widgets=[Label(name='Watershed_segmentation')],labels=True)
        self.vbox.addWidget(segment_tag.native)
        self.vbox.addWidget(segment_cont.native)
        self.vbox.addStretch()

        #make entire widget scrollable
        self.setLayout(self.vbox)
        self.outer_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.outer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.outer_scroll.setWidgetResizable(True)
        self.outer_scroll.setWidget(self)

    #update threshold widget and maxima widget when widget is added to napari GUI
    def eventFilter(self, obj: QObject, event: QEvent):
        if event.type() == QEvent.ParentChange:
            parent = self.parent()
            if isinstance(parent, QDockWidget):
                self.threshold.reset_choices()
                self.maxima.reset_choices()
        return super().eventFilter(obj, event)

