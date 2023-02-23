from qtpy.QtWidgets import QWidget, QMainWindow, QApplication, QDockWidget,QScrollArea, QVBoxLayout,QTabWidget,QPushButton
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget,QVBoxLayout
from qtpy.QtCore import QObject, QEvent, Qt
from magicgui.widgets import LineEdit, SpinBox, Container, PushButton, create_widget, Label
from superqt import QCollapsible

#
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


#widget imports
from ._widgets import Extract_features_widget, Define_class_labels, Train_Classifier, Threshold, Maxima, Segment, Draw

class UIWidget(QWidget):
    """widget overview and place"""
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.outer_scroll = QScrollArea() 
        self.vbox = QVBoxLayout()
        self.installEventFilter(self)

        #organising the final gui
        #instantiate widgets
        self.Extract_features_widget=Extract_features_widget(self)
        self.Define_class_labels_widget=Define_class_labels(self)
        self.Train_Classifier_widget=Train_Classifier(self)
        self.Maxima=Maxima(self)
        self.Threshold=Threshold(self)
        self.Segment=Segment(self)
        self.Draw=Draw(self)

        # add labels
        labels_to_define_tag=Container(widgets=[Label(name='Define_Label_names:')],labels=True)
        label_tag_main=Container(widgets=[Label(name='Layers_to_extract_Features_from:')],labels=True)
        seed_tag_main=Container(widgets=[Label(name='Define_watershed_seeds')],labels=True)
        segment_tag=Container(widgets=[Label(name='Watershed_segmentation')],labels=True)
        draw_tag=Container(widgets=[Label(name='Draw_mother_bud_relations')],labels=True)


        # place widgets in GUI and specify aesthetics.
        #select features widget
        self.vbox.addWidget(label_tag_main.native)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.Extract_features_widget)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.vbox.addWidget(self.scroll)

        #define labels widget
        self.vbox.addWidget(labels_to_define_tag.native)
        self.vbox.addWidget(self.Define_class_labels_widget)

        #classifier widget
        self.vbox.addWidget(self.Train_Classifier_widget)
        
        # add thresholding widgets in collapsible menus
        self.vbox.addWidget(seed_tag_main.native)
        self._collapse1 = QCollapsible('Thresholding', self)
        self._collapse1.addWidget(self.Threshold)
        self.vbox.addWidget(self._collapse1)
        self._collapse2 = QCollapsible('Distance transform:', self)
        self._collapse2.addWidget(self.Maxima)
        self.vbox.addWidget(self._collapse2)


        self.vbox.addWidget(segment_tag.native)
        self.vbox.addWidget(self.Segment)
        self.vbox.addWidget(draw_tag.native)
        self.vbox.addWidget(self.Draw)
        self.vbox.addStretch()

        #make entire widget scrollable
        self.setLayout(self.vbox)
        self.outer_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.outer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.outer_scroll.setWidgetResizable(True)
        self.outer_scroll.setWidget(self)
