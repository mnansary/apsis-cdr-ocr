# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import cv2
import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm.auto import tqdm
tqdm.pandas()
import json
import random

def createHandwritenWords(iden,
                         df,
                         comps):
    '''
        creates handwriten word image
        args:
            iden    :       identifier marking value starting
            df      :       the dataframe that holds the file name and label
            comps   :       the list of components 
        returns:
            img     :       marked word image
            label   :       dictionary of label {iden:label}
            iden    :       the final identifier
    '''
    comps=[str(comp) for comp in comps]
    # select a height
    height=random.randint(32,64)
    # reconfigure comps
    mods=['ঁ', 'ং', 'ঃ']
    while comps[0] in mods:
        comps=comps[1:]
    # construct labels
    label={}
    imgs=[]
    for comp in comps:
        c_df=df.loc[df.label==comp]
        c_df.reset_index(drop=True,inplace=True)
        # select a image file
        idx=random.randint(0,len(c_df)-1)
        img_path=c_df.iloc[idx,0] 
        # read image
        img=cv2.imread(img_path,0)
        # resize
        h,w=img.shape 
        width= int(height* w/h) 
        img=cv2.resize(img,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # mark image
        img=255-img
        data=np.zeros(img.shape)
        data[img>0]      =   iden
        imgs.append(data)
        # label
        label[iden] = comp 
        iden+=1
    img=np.concatenate(imgs,axis=1)
    
    return img,label,iden



def randColor():
    '''
        generates random color
    '''
    d=random.randint(0,64)
    return (d,d,d)

def placeGarbage(img,g,gdf,gcomps):
    xmin,ymin,xmax,ymax=g
    gi,_,_=createHandwritenWords(1,gdf,gcomps)
    gi=cv2.resize(gi,(xmax-xmin,ymax-ymin),fx=0,fy=0,interpolation = cv2.INTER_NEAREST)
    hg,wg=gi.shape
    gw=np.ones((hg,wg),dtype="uint8")*255
    gw[:hg,:wg]=img[ymin:ymax,xmin:xmax]
    gw[gi<255]=randColor()
    img[ymin:ymax,xmin:xmax]=gw
    return img
    

def placeName(iden,df,allcomps,tem,loc_dict,max_wlen=5):
    img=np.copy(tem)
    mask=np.zeros((img.shape[0],img.shape[1]))
    xmin,ymin,xmax,ymax=loc_dict["n"]
    labels=[]
    
    for i in range(random.choice([1,1,1,2,3])):
        comps=[random.choice(allcomps) for _ in range(random.randint(2,max_wlen))]
        name,label,iden= createHandwritenWords(iden,df,comps)
        labels.append(label)
        hw,ww=name.shape
        word=np.ones((hw,ww,3))*255
        word=word.astype("uint8")
        lx=random.randint(xmin,xmax-ww)
        ly=random.randint(ymin,ymax-hw)
        word[:hw,:ww]=img[ly:ly+hw,lx:lx+ww]
        word[name>1]=randColor()
        img[ly:ly+hw,lx:lx+ww]=word
        mask[ly:ly+hw,lx:lx+ww]=name
    return img,mask,labels

def placeNumbers(df,boxes,img):
    for box in boxes:
        xmin,ymin,xmax,ymax=box
        idx=random.randint(0,len(df)-1)
        nimg=cv2.imread(df.iloc[idx,0],0)
        nimg=cv2.resize(nimg,(xmax-xmin,ymax-ymin),fx=0,fy=0,interpolation=cv2.INTER_LINEAR)
        h,w=nimg.shape
        wimg=np.ones((h,w,3),dtype="uint8")*255
        wimg[:h,:w]=img[ymin:ymax,xmin:xmax]
        wimg[nimg==0]=randColor()
        
        img[ymin:ymax,xmin:xmax]=wimg
    return img

def gaussian_heatmap(size=512, distanceRatio=2):
    '''
        creates a gaussian heatmap
        This is a fixed operation to create heatmaps
    '''
    # distrivute values
    v = np.abs(np.linspace(-size / 2, size / 2, num=size))
    # create a value mesh grid
    x, y = np.meshgrid(v, v)
    # spreading heatmap
    g = np.sqrt(x**2 + y**2)
    g *= distanceRatio / (size / 2)
    g = np.exp(-(1 / 2) * (g**2))
    g *= 255
    return g.clip(0, 255).astype('uint8')


def get_maps(cbox,gaussian_heatmap,heat_map,link_map,prev,idx):
    '''
        creates heat_map and link_map:
        args:
            cbox             : charecter bbox[ cxmin,cymin,cxmax,cymax]
            gaussian_heatmap : the original heatmap to fit
            heat_map         : image charecter heatmap
            link_map         : link_map of the word
            prev             : list of list of previous charecter center lines
            idx              : index of current charecter
    '''
    src = np.array([[0, 0], 
                    [gaussian_heatmap.shape[1], 0], 
                    [gaussian_heatmap.shape[1],gaussian_heatmap.shape[0]],
                    [0,gaussian_heatmap.shape[0]]]).astype('float32')

    
    #--------------------
    # heat map
    #-------------------
    cxmin,cymin,cxmax,cymax=cbox
    # char points
    cx1,cx2,cx3,cx4=cxmin,cxmax,cxmax,cxmin
    cy1,cy2,cy3,cy4=cymax,cymax,cymin,cymin
    heat_points = np.array([[cx1,cy1], 
                            [cx2,cy2], 
                            [cx3,cy3], 
                            [cx4,cy4]]).astype('float32')
    M_heat = cv2.getPerspectiveTransform(src=src,dst=heat_points)
    heat_map+=cv2.warpPerspective(gaussian_heatmap,M_heat, dsize=(heat_map.shape[1],heat_map.shape[0]),flags=cv2.INTER_NEAREST).astype('float32')

    #-------------------------------
    # link map
    #-------------------------------
    lx2=cx1+(cx2-cx1)/2
    lx3=lx2
    y_shift=(cy4-cy1)/4
    ly2=cy1+y_shift
    ly3=cy4-y_shift
    if prev is not None:
        prev[idx]=[lx2,lx3,ly2,ly3]
        if idx>0:
            lx1,lx4,ly1,ly4=prev[idx-1]
            link_points = np.array([[lx1,ly1], [lx2,ly2], [lx3,ly3], [lx4,ly4]]).astype('float32')
            M_link = cv2.getPerspectiveTransform(src=src,dst=link_points)
            link_map+=cv2.warpPerspective(gaussian_heatmap,M_link, dsize=(link_map.shape[1],link_map.shape[0]),flags=cv2.INTER_NEAREST).astype('float32')

    return heat_map,link_map,prev

def lineTextPage(page,labels,heatmap):
    '''
        @author
        args:
            page   :     marked image of a page given at letter by letter 
            labels :     list of markings for each word
        returns:
            heatmap,linkmap
         
    '''
    
    # link mask
    link_mask=np.zeros(page.shape)
    # heat mask
    heat_mask=np.zeros(page.shape)
    for label in labels:
        num_char=len(label.keys())
        if num_char>1:
            prev=[[] for _ in range(num_char)]
        else:
            prev=None
        for cidx,(k,v) in enumerate(label.items()):
            if v!=' ':
                idx = np.where(page==k)
                y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
                heat_mask,link_mask,prev=get_maps(  [x_min,y_min,x_max,y_max],
                                                    heatmap,
                                                    heat_mask,
                                                    link_mask,
                                                    prev,
                                                    cidx)
                            
    link_mask=link_mask.astype("uint8")
    heat_mask=heat_mask.astype("uint8")
    return heat_mask,link_mask

def lineTextBox(page,boxes,heatmap):
    # link mask
    link_mask=np.zeros(page.shape)
    # heat mask
    heat_mask=np.zeros(page.shape)
    num_char=len(boxes)
    if num_char>1:
        prev=[[] for _ in range(num_char)]
    else:
        prev=None
    for cidx,box in enumerate(boxes):
        x_min,y_min,x_max,y_max = box
        heat_mask,link_mask,prev=get_maps(  [x_min,y_min,x_max,y_max],
                                            heatmap,
                                            heat_mask,
                                            link_mask,
                                            prev,
                                            cidx)
                    
    link_mask=link_mask.astype("uint8")
    heat_mask=heat_mask.astype("uint8")
    return heat_mask,link_mask

def rotate_image(mat, angle):
    """
        Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),flags=cv2.INTER_NEAREST)
    return rotated_mat

def get_warped_image(img,heat,link,mask,src,warp_vec):
    height,width,_=img.shape
 
    # construct dict warp
    x1,y1=src[0]
    x2,y2=src[1]
    x3,y3=src[2]
    x4,y4=src[3]
    # warping calculation
    xwarp=random.randint(0,30)/100
    ywarp=random.randint(0,30)/100
    # construct destination
    dx=int(width*xwarp)
    dy=int(height*ywarp)
    # const
    if warp_vec=="p1":
        dst= [[dx,dy], [x2,y2],[x3,y3],[x4,y4]]
    elif warp_vec=="p2":
        dst=[[x1,y1],[x2-dx,dy],[x3,y3],[x4,y4]]
    elif warp_vec=="p3":
        dst= [[x1,y1],[x2,y2],[x3-dx,y3-dy],[x4,y4]]
    else:
        dst= [[x1,y1],[x2,y2],[x3,y3],[dx,y4-dy]]
    M   = cv2.getPerspectiveTransform(np.float32(src),np.float32(dst))
    img = cv2.warpPerspective(img, M, (width,height))
    heat= cv2.warpPerspective(heat, M, (width,height),flags=cv2.INTER_NEAREST)
    link= cv2.warpPerspective(link, M, (width,height),flags=cv2.INTER_NEAREST)
    mask= cv2.warpPerspective(mask, M, (width,height),flags=cv2.INTER_NEAREST)
    return img,heat,link,mask,dst

def random_exec(poplutation=[0,1],weights=[0.7,0.3],match=0):
    return random.choices(population=poplutation,weights=weights,k=1)[0]==match

def augment(img,heat,link):
    '''
        augments a base image:
        args:
            img_path   : path of the image to augment
            config     : augmentation config
                         * max_rotation
                         * max_warping_perc
        return: 
            augmented image,augment_mask,augmented_location
    '''
    height,width,d=img.shape
    warp_types=["p1","p2","p3","p4"]
    
    mask=np.ones((height,width))
    curr_coord=[[0,0], 
                [width-1,0], 
                [width-1,height-1], 
                [0,height-1]]
    
    # warp
    if random_exec():
        i=random.choice([0,1])
        if i==0:
            idxs=[0,2]
        else:
            idxs=[1,3]
        if random_exec():    
            idx=random.choice(idxs)
            img,heat,link,mask,curr_coord=get_warped_image(img,heat,link,mask,curr_coord,warp_types[idx])

    if random_exec(): 
        # plane rotation
        angle=random.randint(-30,30)
        img=rotate_image(img,angle)
        heat=rotate_image(heat,angle)
        link=rotate_image(link,angle)
        mask=rotate_image(mask,angle)
        
    return img,heat,link,mask

def backgroundGenerator(_paths,dim=(1024,1024),_type=None):
    '''
    generates random background
    args:
        ds   : dataset object
        dim  : the dimension for background
    '''
    while True:
        if _type is None:
            _type=random.choice(["single","double","comb"])
        if _type=="single":
            img=cv2.imread(random.choice(_paths))
            yield img
        elif _type=="double":
            imgs=[]
            img_paths= random.sample(_paths, 2)
            for img_path in img_paths:
                img=cv2.imread(img_path)
                h,w,d=img.shape
                img=cv2.resize(img,dim)
                imgs.append(img)
            # randomly concat
            img=np.concatenate(imgs,axis=random.choice([0,1]))
            img=cv2.resize(img,(w,h))
            yield img
        else:
            imgs=[]
            img_paths= random.sample(_paths, 4)
            for img_path in img_paths:
                img=cv2.imread(img_path)
                h,w,d=img.shape
                img=cv2.resize(img,dim)
                imgs.append(img)
            seg1=imgs[:2]
            seg2=imgs[2:]
            seg1=np.concatenate(seg1,axis=0)
            seg2=np.concatenate(seg2,axis=0)
            img=np.concatenate([seg1,seg2],axis=1)
            img=cv2.resize(img,(w,h))
            yield img

def padDetectionImage(img,gray=False,pad_value=255):
    cfg={}
    if gray:
        h,w=img.shape
    else:
        h,w,d=img.shape
    if h>w:
        # pad widths
        pad_width =h-w
        # pads
        if gray:
            pad =np.zeros((h,pad_width))
        else:    
            pad =np.ones((h,pad_width,d))*pad_value
        # pad
        img =np.concatenate([img,pad],axis=1)
        # cfg
        cfg["pad"]="width"
        cfg["dim"]=w
    
    elif w>h:
        # pad height
        pad_height =w-h
        # pads
        if gray:
            pad=np.zeros((pad_height,w))
        else:
            pad =np.ones((pad_height,w,d))*pad_value
        # pad
        img =np.concatenate([img,pad],axis=0)
        # cfg
        cfg["pad"]="height"
        cfg["dim"]=h
    else:
        cfg=None
    if not gray:
        img=img.astype("uint8")
    return img
