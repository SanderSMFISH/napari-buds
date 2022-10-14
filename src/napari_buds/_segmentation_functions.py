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
def clean_up(layer):
    layer=remove_small_holes(layer,10)
    layer=erosion(layer,3)
    layer=dilation(layer,3)
    layer=remove_small_objects(layer,min_size=100)

    return layer

#measure distance from cell to blob, currently not used 
def m_dist(cellxy,blobxy):
    xvalue=cellxy[1]-blobxy[1]
    yvalue=cellxy[2]-blobxy[2]
    distance=(xvalue**2+yvalue**2)**.5
            
    return distance

#find nearest nonzero value
def nearest_nonzero(a,x,y):
    idx = np.argwhere(a)
    idx=idx[((idx - [x,y])**2).sum(1).argmin()]
    value=a[idx[0],idx[1]]

    return value

#function that chooses the largest bud and removes the other if more then one bud with the same label value exists 
def continuous_space(buds,original_buds):
    buds_frame=pd.DataFrame(rt(buds,properties=['label','centroid','area']))
    bud_label=np.empty((buds.shape[0],buds.shape[1]),dtype=int)
    for labelid in buds_frame.label.values:
        array=np.where((buds==labelid),original_buds,0)
        rarray=np.ravel(array)
        rarray=np.delete(rarray, np.where(rarray == 0))
        unique_labels=np.unique(rarray)
        if len(unique_labels)>0:
            if len(unique_labels)>1:
                area=dict(Counter(rarray))
                key=max(area, key=area.get)
            else:
                key=unique_labels[0]
            bud_label+=np.where((original_buds==int(key)),buds,0)

    return bud_label

#if multiple cells claim the same bud, this function assigns the bud to the cell with the nearest nuclei 
def contested_buds(buds,original_buds,labels,markers):
    # filter for contested buds
    table_buds=pd.DataFrame(rt(buds,properties=['label','centroid','area']))
    original_budlabels=[original_buds[int(rows[1]),int(rows[2])] for key, rows in table_buds.iterrows()]
    table_buds['original budlabels']=original_budlabels
    dups=table_buds.duplicated('original budlabels')
    table_buds['original bud contested']=dups
    table_buds=table_buds[table_buds['original budlabels']!=0]
    contesting=table_buds[table_buds['original bud contested']==True]
    contested=list(contesting['original budlabels'].values)
    original_buds_frame=pd.DataFrame(rt(original_buds,properties=['label','centroid','area']))
    filter_contest=original_buds_frame.label.isin(contested)
    #budsoi contains only contested buds
    budsoi = original_buds_frame[filter_contest]
    markers=np.where((markers>0),labels,0)
    budsoi=budsoi.set_index('label')
    #find nearest nuclei of contested buds and update label accordingly
    nearest_nuclei=[nearest_nonzero(markers,rows[1][0],rows[1][1]) for rows in budsoi.iterrows()]
    budsoi['nearest nuclei']=nearest_nuclei
    x=np.empty((buds.shape[0],buds.shape[1]),dtype=int)
    for og,rows in budsoi.iterrows():
        decontested_bud=np.where((original_buds==int(og)),int(rows[3]),int(0))
        x=x+decontested_bud 
    corrected_buds=np.where((x>0),x,buds)
    labeled_buds=corrected_buds
    
    return labeled_buds

#segment buds (included bud decontestion and continuos space requirement functions)
def segment_buds(label,result,markers,cell_id,bud_id,bg_id):
    buds=result==bud_id
    buds=clean_up(buds)
    labels_buds,_=nlabel(buds) 
    buds=np.where((label>0) & (buds>0),label,0)
    #budlabels that are contested are linked to closes nuclei
    decontested_buds=contested_buds(buds,labels_buds,label,markers)
    #budlabels can exist in only one continuos location
    labeled_buds_cont=continuous_space(buds,labels_buds)

    return labeled_buds_cont

#cell mask watershed segment function
def watershed_seg(result, seeds,cell_id,bud_id,bg_id):
    cells=result==cell_id
    cells=clean_up(cells)
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
