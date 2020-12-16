# -*- coding: utf-8 -*-
import cv2
import numpy as np
import scipy
from skimage import transform as tf
from skimage.transform import SimilarityTransform
from skimage.transform import matrix_transform
import math
from helpers import interp2
#optical flow functions


def applyGeometricTransformation(features, new_features, bbox, coord, H, W, N):
    """
    Description: Transform bounding box corners onto new image frame
    Input:
        features: Coordinates of all feature points in first frame, (1, N, 2)
        new_features: Coordinates of all feature points in second frame, (1,N, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (1, 2, 2)
    Output:
        features: Coordinates of all feature points in first frame after eliminating outliers, (N, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (1, 2, 2)
    Instruction: Please feel free to use skimage.transform.estimate_transform()
    """
    dist_thresh=20
    #print("shapes of new and origin and mask",new_features.shape, features.shape, mask.shape)
    new_features=new_features.reshape((1,N,-1))
    newFListNum,new_FList=extractFeaturefromFeatures(new_features)
    features=features.reshape((1,N,-1))
    FListNum,FList=extractFeaturefromFeatures(features)
    new_nonZeroFListNum,new_nonZeroFList=extractNonZeroFeature(new_FList)
    nonZeroFListNum,nonZeroFList=extractNonZeroFeature(FList)
    tmp_bbox = np.reshape(bbox,(2,-1))
    idx_range=new_nonZeroFListNum
    #print("Similarity Transform")
    #print("Before transforming: Non zero ",nonZeroFList,"New non zero",new_nonZeroFList, "Bbox:",tmp_bbox)
    transform = SimilarityTransform()
    transformation=transform.estimate(nonZeroFList,new_nonZeroFList)
    
    if transformation:
        homoMatrix=transform.params
        transformed_features = matrix_transform(nonZeroFList,homoMatrix)

    for idx in range(idx_range):
        transformed_point=transformed_features[idx]
        new_point=new_nonZeroFList[idx]
        dist_btw_points=math.sqrt((transformed_point[0]-new_point[0])**2 + (transformed_point[1]-new_point[1])**2)
        if dist_btw_points>dist_thresh:
            new_nonZeroFList[idx,:]=np.array([0,0])

    numArr=np.count_nonzero(coord,axis=0)
    #print("COUNT non zeros",numArr)
    num=np.max(numArr)
    # mask_coords=np.zeros((num,2))
    new_mask_coords=np.zeros((W*H,2))
    # new_mask=np.zeros((mask.shape[0],mask.shape[1]))
    # mask_coords[:,0],mask_coords[:,1]=np.nonzero(mask)
    if transformation:
        new_tmp_bbox=matrix_transform(tmp_bbox,homoMatrix)
        tmp_bbox=new_tmp_bbox
        if tmp_bbox[1][0] > W:
          #print("tmp box range change",tmp_bbox)
          tmp_bbox[1][0]=W
        if tmp_bbox[1][1] > H:
          tmp_bbox[1][1]=H
        if tmp_bbox[0][0] < 0:
          tmp_bbox[0][0]=0
        if tmp_bbox[0][1] < 0:
          tmp_bbox[0][1]=0
        for i in range(num):
           new_mask_coords[i,:]=matrix_transform(coord[i,:],homoMatrix)
           if new_mask_coords[i,0] < tmp_bbox[0][0] or new_mask_coords[i,0]>tmp_bbox[1][0] or new_mask_coords[i,1] < tmp_bbox[0][1] or new_mask_coords[i,1]>tmp_bbox[1][1]:
             new_mask_coords[i,:]=[0,0]
    
    for idx in range(new_nonZeroFListNum):
        new_tmp_bbox_x1=tmp_bbox[0][0]
        new_tmp_bbox_y1=tmp_bbox[0][1]
        new_tmp_bbox_x2=tmp_bbox[1][0]
        new_tmp_bbox_y2=tmp_bbox[1][1]
        if new_nonZeroFList[idx][0] < new_tmp_bbox_x1 or new_nonZeroFList[idx][1]<new_tmp_bbox_y1 or new_nonZeroFList[idx][0]>new_tmp_bbox_x2 or new_nonZeroFList[idx][1]>new_tmp_bbox_y2:
            new_nonZeroFList[idx]=[0,0]
    #print("After Transformation: Nonzero ",new_nonZeroFList,"BBox ", tmp_bbox)
    new_bbox=new_tmp_bbox.reshape(1,2,2)
    features_fillzeros=np.zeros((FListNum,2))

    features_fillzeros[:new_nonZeroFListNum,:]=new_nonZeroFList

    features_fillzeros=features_fillzeros.reshape(FListNum,1,-1)
    return features_fillzeros, new_bbox,new_mask_coords


def estimateAllTranslation(features, img1, img2):
    """
    Description: Get corresponding points for all feature points
    Input:
        features: Coordinates of all feature points in first frame, (F, N, 2)
        img1: First image frame, (H,W)
        img2: Second image frame, (H,W)
    Output:
        new_features: Coordinates of all feature points in second frame, (F, N, 2)
    """
    Ix,Iy=findGradient(img2)
    N=features.shape[0]
    new_features = np.zeros((N,2))
    #print("FEAT",features,features.shape)
    for idx in range(N):
        if features[idx,0]==0 and features[idx,1]==0:
            new_features[idx,:]=features[idx,:]
        else:
            new_features[idx,:]=estimateFeatureTranslation(features[idx], Ix, Iy, img1, img2)
    #print("NEW FEAT",new_features,new_features.shape)
    return new_features.reshape((1,N,2))
    

def estimateFeatureTranslation(feature, Ix, Iy, img1, img2):
    """
    Description: Get corresponding point for one feature point
    Input:
        feature: Coordinate of feature point in first frame, (2,)
        Ix: Gradient along the x direction, (H,W)
        Iy: Gradient along the y direction, (H,W)
        img1: First image frame, (H,W)
        img2: Second image frame, (H,W)
    Output:
        new_feature: Coordinate of feature point in second frame, (2,)
    Instruction: Please feel free to use interp2() and getWinBound() from helpers
    """
    winsize=25
    
    s=(winsize+1)//2
    #print("est feature",feature)
    x=feature[0]
    y=feature[1]
    
    win_l,win_r,win_t,win_b=getWinBound(img1.shape, x, y, winsize)
    
    x_1=np.linspace(win_l,win_r,winsize) #decimal coord patch image
    y_1=np.linspace(win_t,win_b,winsize)
    xx,yy=np.meshgrid(x_1,y_1)
    img1_window=interp2(img1,xx,yy)
    img2_window=interp2(img2,xx,yy)
    dx_sum=0
    dy_sum=0
    Jx,Jy=calcJxJy(Ix,Iy,xx,yy)
    for i in range(50):
        dx,dy=optical_flow(img1_window,img2_window,Jx,Jy)
        dx_sum+=dx
        dy_sum+=dy
        img2_shift=get_new_img(img2,dx_sum,dy_sum)
        img2_window=interp2(img2_shift,xx,yy)

    new_feature = feature + np.array((dx_sum, dy_sum))
    return new_feature


def getFeatures(img,bbox,N):
    """
    Description: Identify feature points within bounding box for each object
    Input:
        img: Grayscale input image, (H, W)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
        N: number of max feature points
    Output:
        number: number of feature points 
        features: Coordinates of all feature points in first frame, (F,N,2)
    Instruction: Please feel free to use cv2.goodFeaturesToTrack() or cv.cornerHarris()
    """
    numOfBboxes=bbox.shape[0]
    features=np.zeros((numOfBboxes,N,2))
    number=[]
    for i in range(numOfBboxes):
      mask = np.zeros(img.shape, dtype=np.uint8)
      mask[int(bbox[i,0,1]):int(bbox[i,1,1]), int(bbox[i,0,0]):int(bbox[i,1,0])] = 255
      #mask=masks[i]
      #mask=np.where(mask==1,255,0)
      mask=mask.astype(np.uint8)
      feature = cv2.goodFeaturesToTrack(img,N,0.3,10, mask=mask)
      if feature is None:
        feature=np.array([0,0,0,0,0,0,0,0,0,0])
        feature=feature.reshape(5,1,2)
        #print("Nonetype returned")
      number.append(feature.shape[0])
      feature=feature.reshape(1,feature.shape[0],2)
      features[i,:feature.shape[1],:]=feature
    #features=features.reshape(numOfBboxes,N,2)
    return number, features
    
def optical_flow(img1, img2, Jx,Jy):
    It = img2 - img1
    A = np.hstack((Jx.reshape(-1, 1), Jy.reshape(-1, 1)))
    b = -It.reshape(-1, 1)
    res = np.linalg.solve(A.T @ A, A.T @ b)
    return res[0, 0], res[1, 0]
    
def getWinBound(img_sz, startX, startY, win_size):
    """
    Description: Generate a window(patch) around the start point
    Input:
        img: Input image 2D shape, (2,)
        startX: start point x coordinate, Scalar
        startY: start point y coordinate, Scalar
        win_size: window size, Scalar
    Output:
        win_left: left bound of window, Scalar
        win_right: right bound of window, Scalar
        win_top: top bound of window, Scalar
        win_bottom: bottom bound of window, Scalar
    """
    szY, szX = img_sz
    
    win_left = startX - (win_size - 1) // 2
    win_right = startX + (win_size + 1) // 2
    if win_left < 0: win_left, win_right = 0, win_size
    elif win_right > szX: win_left, win_right = szX - win_size, szX
        
    win_top = startY - (win_size - 1) // 2
    win_bottom = startY + (win_size + 1) // 2
    if win_top < 0: win_top, win_bottom = 0, win_size
    elif win_bottom > szY: win_top, win_bottom = szY - win_size, szY

    return win_left, win_right, win_top, win_bottom

def get_new_img(img, dx, dy):
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    new_x, new_y = x + dx, y + dy
    return interp2(img, new_x, new_y)

def findGradient(img):
    fx = np.array([[1,0, -1]])
    fy = fx.T
    Ix = scipy.signal.convolve2d(img, fx, 'same', 'symm')
    Iy = scipy.signal.convolve2d(img, fy, 'same', 'symm')
    return Ix, Iy

def calcJxJy(Ix,Iy,xx,yy):
    Jx=interp2(Ix,xx,yy)
    Jy=interp2(Iy,xx,yy)
    return Jx,Jy

def extractFeaturefromFeatures(features):
    featureNum=features.shape[1]
    feature=features.reshape((featureNum,2))
    return featureNum,feature
    
def extractNonZeroFeature(featureList):
    non_zero_mask=np.logical_or(featureList[:,0]!=0,featureList[:,1]!=0)
    nonZero_feature=featureList[non_zero_mask,:]
    nonZeroFeatureNum=nonZero_feature.shape[0]
    return nonZeroFeatureNum,nonZero_feature

def transformMask(initFeatureNum,frame,frame_old,all_featNum,all_features,all_bboxes,all_coords,all_classes,features,old_bbox,old_coord,old_classes,H,W,N):
    #print("Transform mask: feature points",all_features.shape)
    tmp_new_features=estimateAllTranslation(features,frame_old,frame)
    tmp_new_features=np.where(tmp_new_features<0,0,tmp_new_features)
    old_coord=old_coord.reshape(H*W,2)
    tmp_features, bbox, coord = applyGeometricTransformation(features,tmp_new_features,old_bbox,old_coord,H,W,N)
    tmp_features=np.where(tmp_features<0,0,tmp_features)
    add,initFeatureNum,tmp_features,bbox, eraseObject=generateMoreFeatures(initFeatureNum,frame, tmp_features,old_bbox,bbox,W,H,N)
    if add==True:
      #print("ALLFEAT",all_features.shape, tmp_features.shape)
      all_features=np.append(all_features,tmp_features.reshape(1,N,2))
      all_bboxes=np.append(all_bboxes,bbox)
      all_coords=np.append(all_coords,coord)
      all_classes.append(old_classes)
      all_featNum.append(initFeatureNum)
    return all_featNum,all_features,all_bboxes,coord,all_coords,all_classes, eraseObject

def generateMoreFeatures(initFeatureNum,frame,new_features,old_bbox,bbox,W,H,N):
  """
  delete bbox if the object disappears 
  if the features are less than the 60% of the initial number of features, new features will be generated
  """
  new_FListNum,new_FList=extractFeaturefromFeatures(new_features.reshape(1,-1,2))
  remainNumOfFList, remainFList=extractNonZeroFeature(new_FList)
  old_bbox=old_bbox.reshape(1,2,2)
  bbox=bbox.reshape(2,2)
  x=initFeatureNum
  eraseObject=False
  #print("feature number",initFeatureNum)
  if remainNumOfFList < initFeatureNum * 0.6:
      #print("****** bbox lost****")
      eraseObject=True
      #temporary: not going to generate new bounding box
      return False, x, new_features,bbox, eraseObject
      #print("BBOX shape,",bbox.shape)
      bbox_w=bbox[1,0]-bbox[0,0]
      bbox_h=bbox[1,1]-bbox[0,1]
      #print("BBOX\n",bbox, bbox_w,bbox_h)
      if bbox_w<10 or bbox_h <10:
        #print("bbox too small")
        return False, x, new_features, bbox, eraseObject
      elif bbox[1,0]==W or bbox[1,1]==H:
        #print("bbox out of bound")
        return False, x, new_features, bbox, eraseObject
      elif bbox[0,0]==0 or bbox[0,1]==0:
        #print("bbox out of bound")
        return False, x, new_features, bbox, eraseObject
      #use old bbox to generate new features
      x,new_features = getFeatures(frame, old_bbox,N)

  return True, x, new_features, bbox, eraseObject

def generateMaskWithCoordinates(mask_coords,W,H):
  """
  generate mask (H,W) by filling ones to given mask coordinates 
  """
  new_mask=np.zeros((H,W))
  mask_coords=mask_coords.reshape(-1,2)
  numArr=np.count_nonzero(mask_coords,axis=0)
  num=np.max(numArr)
  for i in range(num):
    if (mask_coords[i,0]>0 and mask_coords[i,0]<=W and mask_coords[i,1]>0 and mask_coords[i,1]<=H):
      new_mask[int(mask_coords[i,1]),int(mask_coords[i,0])]=1
  return new_mask
  
  
# def generateAllMasksWithCoordinates(all_mask_coords,W,H):
#   """
#   generate all masks (H,W) by filling ones to given mask coordinates 
#   """
#   numOfmasks=all_mask_coords.shape[0]
#   print("ALL coords",all_mask_coords.shape)
#   masks=[]
#   for i in range(numOfmasks):
#       new_mask=np.zeros((H,W))
#       mask_coords=all_mask_coords[i]
#       mask_coords=mask_coords.reshape(-1,2)
#       numArr=np.count_nonzero(mask_coords,axis=0)
#       num=np.max(numArr)
#       for i in range(num):
#         if (mask_coords[i,0]>0 and mask_coords[i,0]<=W and mask_coords[i,1]>0 and mask_coords[i,1]<=H):
#           new_mask[int(mask_coords[i,1]),int(mask_coords[i,0])]=1
#       masks.append(new_mask)
#   masks=np.array(masks)
#   masks=masks.reshape(numOfmasks,H,W)
#   return masks

def generateCoordinatesOfMask(mask,W,H):
  """
  generate coordinates x,y of mask that are not zero
  """
  num=np.count_nonzero(mask)
  mask_coords=np.zeros((W*H,2))
  mask_coords[:num,1],mask_coords[:num,0]=np.nonzero(mask)
  return mask_coords

def generateAllCoordinates(masks, W,H):
  """
    generate coordinates multiple masks
  """
  num=masks.shape[0]
  coords=np.array([])
  for i in range(num):
    coords=np.append(coords,generateCoordinatesOfMask(masks[i],W,H))
  coords=coords.reshape(num,H*W,2)
  return coords

# def transformMask2(initFeatureNum,frame,frame_old,all_featNum,all_features,all_bboxes,all_coords,all_classes,features,old_bbox,old_coord,old_classes,H,W,N, LKparam):
#     print("Transform mask: feature points",features)
#     #tmp_new_features=estimateAllTranslation(features,frame_old,frame)
#     # frame_old=np.asarray(frame_old,dtype="float32")
#     # frame=np.asarray(frame,dtype="float32")
#     features=features.astype(np.float32)
#     tmp_new_features, st, err = cv2.calcOpticalFlowPyrLK(frame_old.astype(np.uint8), frame.astype(np.uint8), features.reshape(N,1,2), None, **LKparam)
#     good_new = tmp_new_features[st==1]
#     print("GOOD NEW FEATURES:",tmp_new_features)
#     print("ST",st)
#     print("ERR",err)
#     if good_new.shape[0]==0:
#       tmp_new_features=features
#     else:
#       tmp_new_features=good_new.reshape(1,N,2)
#     tmp_new_features=np.where(tmp_new_features<0,0,tmp_new_features)
#     old_coord=old_coord.reshape(H*W,2)
#     tmp_features, bbox, coord = applyGeometricTransformation(features,tmp_new_features,old_bbox,old_coord,H,W,N)
#     tmp_features=np.where(tmp_features<0,0,tmp_features)
#     add,initFeatureNum,tmp_features,bbox, eraseObject=generateMoreFeatures(initFeatureNum,frame, tmp_features,old_bbox,bbox,W,H)
#     if add==True:
#       print("ALLFEAT",all_features.shape, tmp_features.shape)
#       all_features=np.append(all_features,tmp_features.reshape(1,N,2))
#       all_bboxes=np.append(all_bboxes,bbox)
#       all_coords=np.append(all_coords,coord)
#       all_classes.append(old_classes)
#       all_featNum.append(initFeatureNum)
#     return all_featNum,all_features,all_bboxes,coord,all_coords,all_classes, eraseObject

