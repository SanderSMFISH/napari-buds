from magicgui.widgets import LineEdit, SpinBox, Container, PushButton, create_widget, Label
from magicgui import magicgui,magic_factory
from magicclass import HasFields, vfield, magicclass, set_design, set_options
from magicclass.types import OneOf
from magicgui.widgets import LineEdit, SpinBox, Container, PushButton, create_widget, Label, ComboBox, Slider, Label,FileEdit
from qtpy.QtWidgets import (QWidget,QVBoxLayout,QTabWidget,QPushButton,QHBoxLayout)
from superqt import QCollapsible
from magicgui import widgets, magicgui
from napari.layers import Image, Labels
from qtpy.QtWidgets import QWidget, QMainWindow, QApplication, QDockWidget,QScrollArea, QPushButton
from qtpy.QtCore import QObject, QEvent, Qt
from ._segmentation_functions import dilation,erosion,watershed_seg, clean_up, label_id, draw_mother_bud_relations, count_class_labels
import joblib
import numpy as np
from scipy.ndimage import label as nlabel
from scipy.ndimage import distance_transform_edt as distance
from functools import partial
from sklearn.ensemble import RandomForestClassifier
from skimage import io,segmentation,data, feature, future
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from superqt import QLabeledRangeSlider,QRangeSlider
from superqt import QCollapsible


##################################################################################################################################################

class Extract_features_widget(QWidget):
    """widget to extract features from image layers"""
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent 
        self.viewer = parent.viewer
        self.setLayout(QHBoxLayout())

        self.hidden_layers=['result','seeds','Labels','relations mother buds','cell mask','distance','buds']

        #extract layer names for feature extraction
        labels_FE=[self.viewer.layers[i].name for i in range(len(self.viewer.layers))][::-1]
        labels_FE=[x for x in labels_FE if x not in self.hidden_layers]
        self.labels_FE=labels_FE
        self.layers_to_select = Container(widgets=[create_widget(name=label, widget_type='CheckBox',value=1) for label in labels_FE])

        #update layer choices after changes to label layer
        @self.viewer.layers.events.connect
        def update_layer_extraction_container():
            labels_FE=[self.viewer.layers[i].name for i in range(len(self.viewer.layers))][::-1]
            labels_FE=[x for x in labels_FE if x not in self.hidden_layers]
            old_layers=self.layers_to_select.asdict()

            #update selectable layers widget
            for layer in old_layers.keys():
                if layer not in labels_FE:
                    self.layers_to_select.remove(layer)
            i=1
            for layer in labels_FE:
                if layer not in old_layers:
                    self.layers_to_select.insert(i,create_widget(name=layer, widget_type='CheckBox'))
                i+=1
            self.labels_FE=labels_FE

        self.layout().addWidget(self.layers_to_select.native)

####################################################################################################################################################

class Define_class_labels(QWidget):
    """widget to define class label names for each label"""
    def __init__(self,parent):
        super().__init__()

        self.parent = parent 
        self.viewer = parent.viewer
        self.setLayout(QVBoxLayout())

        # set default class labels
        self.class_labels =  ['cells','buds','background']
        slabels=self.class_labels
        try:
            widget_class_names = count_class_labels(self.viewer.layers['Labels'].data)
        except:
            widget_class_names = ['class_1','class_2','class_3']

        #add numbered class labels if exceeds defeault
        if len(widget_class_names)>len(slabels) & len(widget_class_names)!=0:
            i=len(widget_class_names)
            j=len(slabels)
            slabels = slabels + widget_class_names[j:i]

        #add only necessary number of default class labels
        if len(widget_class_names)<len(slabels):
            i=len(widget_class_names)
            j=len(slabels)
            slabels = slabels[:i]
        
        #save resulting label names
        self.class_labels=slabels

        #create container widget with label names
        self.labels_to_define = Container(widgets=[create_widget(slabel,name=widget_name) for widget_name,slabel in zip(widget_class_names,slabels)])

        #PushButton to refresh class labels that are defined in the GUI
        Refresh_labels=PushButton(name="Refresh")
        Refresh_labels.is_connected = False

        
        #updat class labels when refresh button is presesed
        @Refresh_labels.changed.connect
        def _on_connect():

            #change button text
            Refresh_labels.text = 'Refreshed' if Refresh_labels.is_connected else 'Refresh'

            #redefine class labels
            slabels =  self.class_labels
            pot_widget_names = count_class_labels(self.viewer.layers['Labels'].data)
            
            if len(pot_widget_names)!=len(slabels):
                i=len(pot_widget_names)
                j=len(slabels)
                slabels = slabels + pot_widget_names[j:i]

                for widget_name,slabel, in self.labels_to_define.asdict().items():
                    if widget_name not in pot_widget_names:
                        self.labels_to_define.remove(widget_name)
                for pot_widget_name in pot_widget_names:
                    if pot_widget_name not in self.labels_to_define.asdict().keys():
                        self.labels_to_define.append(create_widget(pot_widget_name,name=pot_widget_name))

            # reset button
            Refresh_labels.is_connected = False
            Refresh_labels.text = 'Refreshed' if Refresh_labels.is_connected else 'Refresh'

        self.layout().addWidget(self.labels_to_define.native)
        self.layout().addWidget(Refresh_labels.native)

####################################################################################################################################################
@magic_factory(call_button="save settings")
def RF_settings(intensity=True, edges=True, texture=True,sigma_min=1, sigma_max=20, n_estimators=100,n_jobs=-1,max_depth=10,max_samples=0.05):
    print('updated settings')
    

class Train_Classifier(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.viewer=parent.viewer
        self.setLayout(QVBoxLayout())

        self.settings = RF_settings()
        self.features_func=partial(feature.multiscale_basic_features,
                            intensity=self.settings.intensity.value, edges=self.settings.edges.value, texture=self.settings.texture.value,
                            sigma_min=self.settings.sigma_min.value, sigma_max=self.settings.sigma_max.value)
        self.rf_params=RandomForestClassifier(n_estimators=self.settings.n_estimators.value, n_jobs=self.settings.n_jobs.value,
                        max_depth=self.settings.max_depth.value, max_samples=self.settings.max_samples.value)

        def save_settings():
            self.features_func=partial(feature.multiscale_basic_features,
                            intensity=self.settings.intensity.value, edges=self.settings.edges.value, texture=self.settings.texture.value,
                            sigma_min=self.settings.sigma_min.value, sigma_max=self.settings.sigma_max.value)
            self.rf_params=RandomForestClassifier(n_estimators=self.settings.n_estimators.value, n_jobs=self.settings.n_jobs.value,
                        max_depth=self.settings.max_depth.value, max_samples=self.settings.max_samples.value)
            print('save settings')
            print(self.features_func)


        #train classifier by extracting from checked feature layers and fitting + predicting random forest parameters
        self.train_button=PushButton(label='Train classifier')

        def train_classifier():
            fs_features=self.parent.Extract_features_widget.layers_to_select.asdict()
            fs=[]
            for fs_feature,check in fs_features.items():
                if check==True:
                    array=self.viewer.layers[str(fs_feature)].data.astype(np.uint16)
                    fs.append(self.features_func(array))
            features = np.concatenate(fs, axis=-1)
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
        self.classify_button=PushButton(label="Classify")

        def classify():
            fs_features=self.parent.Extract_features_widget.layers_to_select.asdict()
            fs=[]
            for fs_feature,check in fs_features.items():
                if check==True:
                    array=self.viewer.layers[str(fs_feature)].data.astype(np.uint16)
                    fs.append(self.features_func(array))
            features = np.concatenate(fs, axis=-1)
            clf = self.clf
            #training_labels = self.viewer.layers['Labels'].data.astype(np.uint32)
            result = future.predict_segmenter(features, clf)
            try:
                self.viewer.remove('result')
            except:
                pass
            self.viewer.add_labels(result,name='result',opacity=0.5) 

        #load classifier
        self.load_edit=FileEdit(label="Load classifier", mode='r')
        self.load_button=PushButton(label="Load")
        self.load_container=Container(widgets=[self.load_edit,self.load_button],layout='horizontal')

        def load_classifier():
            try:
                file=self.load_edit.value
                self.clf=joblib.load(str(file))
            except:
                self.viewer.status='classifier not loaded'

        #load classifier
        self.save_edit=FileEdit(label="Save classifier", mode='w', value='file')
        self.save_button=PushButton(label="Save")
        self.save_container=Container(widgets=[self.save_edit,self.save_button],layout='horizontal')

        def save_classifier():
            file=self.save_edit.value
            print(file)
            joblib.dump(self.clf,str(file))
            try:
                file=self.load_edit.value
                joblib.dump(self.clf,str(file))
            except:
                self.viewer.status='classifier not saved'

        #connect buttons and functions
        self.train_button.changed.connect(train_classifier)
        self.classify_button.changed.connect(classify)
        self.load_button.changed.connect(load_classifier)#add label horizontal
        self.save_button.changed.connect(save_classifier)#add label horizontal
        self.settings.call_button.clicked.connect(save_settings)

        #create container with classifier settings
        self._collapse1 = QCollapsible('RF settings', self)
        self._collapse1.addWidget(self.settings.native)
        self.cont_Train_Classifier=Container(widgets=[self.train_button,self.classify_button,self.load_container,self.save_container],labels=False)

        #add container to widget layout
        self.layout().addWidget(self._collapse1)
        self.layout().addWidget(self.cont_Train_Classifier.native)

###################################################################################################################################################

#define seeds for watershed segmentation by thresholding
class Threshold(QWidget):
    def __init__(self,parent):
        super().__init__()

        self.parent = parent 
        self.viewer = parent.viewer
        self.setLayout(QHBoxLayout())

        #select image layer
        self.input =ComboBox(
            label="Image",
            annotation=Image,
            choices=[
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, Image)
            ],
        )
        self.to_threshold_img=self.viewer.layers[self.input.current_choice].data

        #update possible selections
        @self.viewer.layers.events.connect
        def update_images():
            self.input.choices=[
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, Image)
            ]

        @self.input.changed.connect
        def update_current_choice():
            self.to_threshold_img=self.viewer.layers[self.input.current_choice].data

        self.threshold_slider = Slider(value=0, min=0,max=100, label=f'Threshold',tracking=False)

        @self.threshold_slider.changed.connect
        def create_seeds():
            scaled_img = self.to_threshold_img*(100/self.to_threshold_img.max())
            threshold = self.threshold_slider.value
            seeds = scaled_img > threshold
            seeds = clean_up(seeds,10)
            markers,_= nlabel(seeds)
            try:
                self.viewer.layers.remove('seeds')
            except:
                pass
            self.viewer.add_labels(markers,name='seeds')

        #add image selection widget
        self.layout().addWidget(Container(widgets=[self.input,self.threshold_slider],layout='vertical').native)

####################################################################################################################################################

  #define seeds by peak local maxima on distance transformed image
class Maxima(QWidget):
    """widget to define class label names for each label"""
    def __init__(self,parent):
        super().__init__()

        self.parent = parent 
        self.viewer = parent.viewer
        self.setLayout(QVBoxLayout())

                #select image layer
        self.input =ComboBox(
            label="Image",
            annotation=Image,
            choices=[
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, Image)
            ],
        )
        self.to_threshold_img= self.viewer.layers[self.input.current_choice].data

        #update possible selections
        @self.viewer.layers.events.connect
        def update_images():
            self.input.choices=[
                layer.name
                for layer in self.viewer.layers
                if isinstance(layer, Image)
            ]

        #update when input is changed
        @self.input.changed.connect
        def update_current_choice():
            self.to_threshold_img=self.viewer.layers[self.input.current_choice].data

        #create widget for editing minimal distance between peaks
        self.min_distance=LineEdit(value=10, label='Minimal distance')
        self.update_distances=PushButton(label='Update')
        self.container=Container(widgets=[self.min_distance,self.update_distances],layout='horizontal')

        #add sliders for thresholding local peaks
        self.local_peaks_slider = Slider(value=0, min=0,max=100, label=f'Find local Maxima',tracking=False)
        self.rel_threshold_slider = Slider(value=0, min=0,max=100, label=f'Relative threshold',tracking=False)

        #calculate local peaks
        def create_seeds():
            scaled_img = self.to_threshold_img*(100/self.to_threshold_img.max())
            threshold = self.local_peaks_slider.value
            thresholded_image=scaled_img>threshold
            distance_image=distance(thresholded_image)
            local_max_coords = feature.peak_local_max(
            distance_image,
            min_distance=int(self.min_distance.value),
            threshold_rel=int(self.rel_threshold_slider.value)/100,
            threshold_abs=None,
            )
            local_max_mask = np.zeros(distance_image.shape, dtype=bool)
            local_max_mask[tuple(local_max_coords.T)] = True
            local_max_mask=dilation(local_max_mask,3)
            markers, _ = nlabel(local_max_mask)
            try:
                self.viewer.layers.remove('seeds')
                self.viewer.layers.remove('distance')
            except:
                pass
            self.viewer.add_image(distance_image,name='distance',opacity=0.80)
            self.viewer.add_labels(markers,name='seeds')
        
        #update local peaks when changes to slider are made        
        self.local_peaks_slider.changed.connect(create_seeds)
        self.rel_threshold_slider.changed.connect(create_seeds)
        self.input.changed.connect(create_seeds)
        self.update_distances.changed.connect(create_seeds)
        
        #add widgets to Qwidget     
        self.layout().addWidget(Container(widgets=[self.input,self.local_peaks_slider,self.rel_threshold_slider],layout='vertical').native)       
        self.layout().addWidget(self.container.native)

# ##################################################################################################################################################

class Segment(QWidget):
    """widget to define class label names for each label"""
    def __init__(self,parent):
        super().__init__()

        self.parent = parent 
        self.viewer = parent.viewer
        self.setLayout(QVBoxLayout())

        self.labels_to_define=self.parent.Define_class_labels_widget.labels_to_define

        @self.parent.Define_class_labels_widget.labels_to_define.changed.connect
        def update_labels_to_define():
            self.labels_to_define=self.parent.Define_class_labels_widget.labels_to_define

        self.segment_button=PushButton(label="Segment")

        def segment_cells():
            cell_id,bud_id,bg_id=label_id(self.labels_to_define.asdict())
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

        self.segment_button.changed.connect(segment_cells)

        self.layout().addWidget(self.segment_button.native)

####################################################################################################################################################

class Draw(QWidget):
    """widget to draw mother bud connections between bud and cell mask layer"""
    def __init__(self,parent):
        super().__init__()

        self.parent = parent 
        self.viewer = parent.viewer
        self.setLayout(QVBoxLayout())

        self.draw_button=PushButton(label="Draw")

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

        self.draw_button.changed.connect(draw_mother_bud)
        self.layout().addWidget(self.draw_button.native)

####################################################################################################################################################