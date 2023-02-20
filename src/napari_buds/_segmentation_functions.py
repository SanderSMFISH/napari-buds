from skimage.morphology import binary_dilation,binary_erosion
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt as distance
from skimage.morphology import remove_small_holes,remove_small_objects
from skimage.segmentation import watershed
from scipy.ndimage import label as nlabel
from skimage.measure import regionprops_table as rt
from collections import Counter
from skimage.draw import line_aa

#defining the number of classes drawn in the Labels layer of the viewer
def count_class_labels(labels):
    labels=np.unique(labels)
    labels=list(np.delete(labels, np.where(labels == 0)))
    labels=[f'class_{label}' for label in labels]

    return labels

#simple iterative erosion function for removing stray classification or thresholded seeds
def erosion(img, iter):
    for i in range(0,iter):
        img=binary_erosion(img)
    return img

#simple iterative erosion function for merging stray classification or thresholded seeds
def dilation(img, iter):
    for i in range(0,iter):
        img=binary_dilation(img)
    return img

#function for cleaning up segmentation errors 
def clean_up(layer,minimal_object_size):
    layer=remove_small_holes(layer,10)
    layer=erosion(layer,3)
    layer=dilation(layer,3)
    layer=remove_small_objects(layer,min_size=minimal_object_size)

    return layer

def nearest_nonzero_label(a,y,x):
    x,y=int(x),int(y)
    #minimal area to look for new labels
    minimal_area=35
    #y values 
    ymin=y-minimal_area
    ymax=y+minimal_area
    #x values
    xmin=x-minimal_area
    xmax=x+minimal_area
    
    if xmin<0:
        xmin=0
    if ymin<0:
        ymin=0

    a=a[ymin:ymax,xmin:xmax]
    idx = np.argwhere(a)
    
    if len(idx)>0:
        idx=idx[((idx - [y,x])**2).sum(1).argmin()]
        new_label=a[idx[0],idx[1]]
    else:
        new_label=0

    return new_label

#function that chooses the largest bud and removes the other if more then one bud with the same label value exists 
def continuous_space(new_buds,old_buds):
    buds_frame=pd.DataFrame(rt(new_buds,properties=['label','centroid','area']))
    for labelid in buds_frame.label.values:
        array=np.where((new_buds==labelid),old_buds,0)
        rarray=np.ravel(array)
        rarray=rarray[rarray!=0]
        unique_labels=np.unique(rarray)
        if len(unique_labels)>1:
            #find the largest area in the orignal bud layer
            area=dict(Counter(rarray))
            #get its label
            key=max(area, key=area.get)
            smaller_bud_labels=unique_labels[unique_labels!=key]
            #remove from the orignal budlayer the smaller area
            new_buds=np.where(np.isin(old_buds,smaller_bud_labels),0,new_buds)

    return new_buds

#if multiple cells claim the same bud, this function assigns the bud to the cell with the largest area
def contested_buds(new_buds,old_buds):
    # filter for contested buds
    old_buds_id=np.unique(old_buds)
    old_buds_id=old_buds_id[old_buds_id!=0]
    ROIs=[np.where(old_buds==old_id,new_buds,0) for old_id in old_buds_id]

    # find new label buds contained in old buds (ROIs)
    frames=[pd.DataFrame(rt(ROI,properties=['label','centroid','area'])) for ROI in ROIs]
    # find the largest bud label in a old bud label
    largest_labels=[frame.sort_values(by='area',ascending=False).label.values[0] for frame in frames if len(frame)>0]
    # find all other labels
    all_other_labels=[frame.sort_values(by='area',ascending=False).label.values[1:] for frame in frames if len(frame)>0]
    # replace all other labels with largest label
    for largest_label,other_labels in zip(largest_labels,all_other_labels):
        new_buds=np.where(np.isin(new_buds,other_labels),largest_label,new_buds)
    
    return new_buds 

def assign_unlabelled_buds(new_buds,old_buds,labels):
    # find buds that have not been assigned a label
    unlabelled_buds=np.where((old_buds>0) & (new_buds==0),old_buds,0)
    # assign new labels based on the closest cell mask to these buds
    buds_frame=pd.DataFrame(rt(unlabelled_buds,properties=['label','area','centroid']))
    for index,row in buds_frame.iterrows():
        new_label=nearest_nonzero_label(labels,row['centroid-0'],row['centroid-1'])
        new_buds=np.where((old_buds==row['label'])&(~np.isin(new_buds,new_label)),new_label,new_buds) 
        
    return new_buds

#segment buds (included bud decontestion and continuos space requirement functions)
def segment_buds(label,result,markers,cell_id,bud_id,bg_id):
    labeled_buds_cont=np.zeros((result.shape[0],result.shape[1]),dtype=int)
    if bud_id!=None:
        buds=result==bud_id
        buds=clean_up(buds,10)
        labels_buds,_= nlabel(buds) 
        # flood everywhere where label and bud agree with congruent bud and cell labels
        buds=np.where((label>0) & (buds>0),label,0)
        #assign buds that aren't assigned to a cell to closest cell mask
        labeled_buds_assigned=assign_unlabelled_buds(buds,labels_buds,label)
        #budlabels that are contested are given to the largest nuclei
        decontested_buds=contested_buds(labeled_buds_assigned,labels_buds)
        #budlabels can exist in only one continuos location
        labeled_buds_cont=continuous_space(decontested_buds,labels_buds)

    return labeled_buds_cont

#cell mask watershed segment function
def watershed_seg(result, seeds,cell_id,bud_id,bg_id):
    cells=result==cell_id
    cells=clean_up(cells,100)
    markers=seeds
    non_background=np.where((result > 0) & (result!=bg_id),result,0)
    non_background=non_background>0
    dist=distance(non_background).astype(int)
    label=watershed(-dist,markers,mask=non_background)
    labeled_buds=segment_buds(label,result,markers,cell_id,bud_id,bg_id)
    label=np.where((labeled_buds>0)&(label!=labeled_buds),labeled_buds,label)
        
    return label, labeled_buds

#reads the define label classes and 
def label_id(dictionary_of_class_labels):

    cell_id,bud_id,bg_id= None, None, None

    for k, v in dictionary_of_class_labels.items():
        if v=='buds':
            bud_id=int(str(k).split('_')[1])
        if v=='cells':
            cell_id=int(str(k).split('_')[1])
        if v=='background':
            bg_id=int(str(k).split('_')[1])

    return cell_id,bud_id,bg_id

#draw mother buds relations in order to inspect correctness
def draw_mother_bud_relations(label, labeled_buds):
    vector_plot = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint32)
    table_buds=pd.DataFrame(rt(labeled_buds,properties=['label','centroid'])).rename(columns={"label":"label","centroid-0": "bud_x","centroid-1":"bud_y"})
    table_cells=pd.DataFrame(rt(label,properties=['label','centroid'])).rename(columns={"label":"label","centroid-0":"cell_x","centroid-1":"cell_y"})
    if len(table_buds)!=0 or len(table_cells)!=0:
        filt_table_cells=table_cells[table_cells['label'].isin(table_buds['label'])] 
        merged=table_buds.merge(filt_table_cells,left_on='label', right_on='label')

        merged=merged.set_index('label')
        for key,value in merged.iterrows():
            rr, cc, val = line_aa(int(value['bud_x']),int(value['bud_y']), int(value['cell_x']), int(value['cell_y']))
            vector_plot[rr, cc] = val * 255 #add the vectors

    return label, labeled_buds, vector_plot
