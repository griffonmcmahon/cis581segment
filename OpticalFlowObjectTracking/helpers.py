import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import signal
from PIL import Image
import argparse
import cv2

def interp2(v, xq, yq):
	dim_input = 1
	if len(xq.shape) == 2 or len(yq.shape) == 2:
		dim_input = 2
		q_h = xq.shape[0]
		q_w = xq.shape[1]
		xq = xq.flatten()
		yq = yq.flatten()

	h = v.shape[0]
	w = v.shape[1]
	if xq.shape != yq.shape:
		raise 'query coordinates Xq Yq should have same shape'

	x_floor = np.floor(xq).astype(np.int32)
	y_floor = np.floor(yq).astype(np.int32)
	x_ceil = np.ceil(xq).astype(np.int32)
	y_ceil = np.ceil(yq).astype(np.int32)

	x_floor[x_floor < 0] = 0
	y_floor[y_floor < 0] = 0
	x_ceil[x_ceil < 0] = 0
	y_ceil[y_ceil < 0] = 0

	x_floor[x_floor >= w-1] = w-1
	y_floor[y_floor >= h-1] = h-1
	x_ceil[x_ceil >= w-1] = w-1
	y_ceil[y_ceil >= h-1] = h-1

	v1 = v[y_floor, x_floor]
	v2 = v[y_floor, x_ceil]
	v3 = v[y_ceil, x_floor]
	v4 = v[y_ceil, x_ceil]

	lh = yq - y_floor
	lw = xq - x_floor
	hh = 1 - lh
	hw = 1 - lw

	w1 = hh * hw
	w2 = hh * lw
	w3 = lh * hw
	w4 = lh * lw

	interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

	if dim_input == 2:
		return interp_val.reshape(q_h, q_w)
	return interp_val

def Test_script(I, E):
    test_pass = True

    # E should be 2D matrix
    if E.ndim != 2:
      print('ERROR: Incorrect Edge map dimension! \n')
      print(E.ndim)
      test_pass = False
    # end if

    # E should have same size with original image
    nr_I, nc_I = I.shape[0], I.shape[1]
    nr_E, nc_E = E.shape[0], E.shape[1]

    if nr_I != nr_E or nc_I != nc_E:
      print('ERROR: Edge map size has changed during operations! \n')
      test_pass = False
    # end if

    # E should be a binary matrix so that element should be either 1 or 0
    numEle = E.size
    numOnes, numZeros = E[E == 1].size, E[E == 0].size

    if numEle != (numOnes + numZeros):
      print('ERROR: Edge map is not binary one! \n')
      test_pass = False
    # end if

    if test_pass:
      print('Shape Test Passed! \n')
    else:
      print('Shape Test Failed! \n')

    return test_pass

#canny edge detection functions
def cannyEdge(I, low, high):
    # convert RGB image to gray color space
    im_gray = rgb2gray(I)

    Mag, Magx, Magy, Ori = findDerivatives(im_gray)
    M = nonMaxSup(Mag, Ori)
    E = edgeLink(M, Mag, Ori, low, high)

    # only when test passed that can show all results
    if Test_script(im_gray, E):
        # visualization results
        print("passed the tes")

    return E

def rgb2gray(I_rgb):
    r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
    I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return I_gray

def findDerivatives(I_gray):
    '''
    File clarification:
        Compute gradient information of the input grayscale image
        - Input I_gray: H x W matrix as image
        - Output Mag: H x W matrix represents the magnitude of derivatives
        - Output Magx: H x W matrix represents the magnitude of derivatives along x-axis
        - Output Magy: H x W matrix represents the magnitude of derivatives along y-axis
        - Output Ori: H x W matrix represents the orientation of derivatives
    '''
    G=np.array([[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]])
    G=G/159
    dx=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    dy=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    Gx=signal.convolve2d(G,dx,'same')
    Gy=signal.convolve2d(G,dy,'same')
    lx=signal.convolve2d(I_gray,Gx,'same')
    ly=signal.convolve2d(I_gray,Gy,'same')
    
    Mag = ((lx**2)+(ly**2))**(1/2)
    Ori = np.arctan2(ly,lx)
    Magx = lx
    Magy = ly
    return Mag,Magx,Magy,Ori

def nonMaxSup(Mag, Ori):
    '''
    File clarification:
        Find local maximum edge pixel using NMS along the line of the gradient
        - Input Mag: H x W matrix represents the magnitude of derivatives
        - Input Ori: H x W matrix represents the orientation of derivatives
        - Output M: H x W binary matrix represents the edge map after non-maximum suppression
    '''
    # getting neighbor in the gradient oriention direction
    nc,nr=Mag.shape[1],Mag.shape[0]
    x,y=np.meshgrid(np.arange(nc),np.arange(nr))
    right=x+np.cos(Ori)
    down=y+np.sin(Ori)
    r_neighbor=np.clip(right,0,nc-1)
    d_neighbor=np.clip(down,0,nr-1)
    neighbor=interp2(Mag,right,down)
    # getting neighbor in the opposite of the oritention direction
    left=x-np.cos(Ori)
    up=y-np.sin(Ori)
    l_neighbor=np.clip(left,0,nc-1)
    up_neighbor=np.clip(up,0,nr-1)
    op_neighbor=interp2(Mag,left,up)
    # perform NMS
    bound=(right<0) | (right>nc-1) | (down<0) | (down>nr-1)
    op_bound=(left<0) | (left>nc-1) | (up<0) | (up>nr-1)
    neighbor[bound]=0
    op_neighbor[op_bound]=0
    nms=np.logical_and(neighbor<Mag,op_neighbor<Mag) 
    return nms

def edgeLink(M, Mag, Ori, low, high):
    '''
    File clarification:
        Use hysteresis to link edges based on high and low magnitude thresholds
        - Input M: H x W logical map after non-max suppression
        - Input Mag: H x W matrix represents the magnitude of gradient
        - Input Ori: H x W matrix represents the orientation of gradient
        - Input low, high: low and high thresholds 
        - Output E: H x W binary matrix represents the final canny edge detection map
    '''
    # suppress pixels whose magnitude is lower than low threshold
    #why use M when it's not same as Mag with NMS, use M for both weak edge (weakEdgeMap) and initial strong edge map (edgeMap)
    #M is the resulf of NMS we are going to use that instead of calling NMS function
    nc,nr=Mag.shape[1],Mag.shape[0]
    x,y=np.meshgrid(np.arange(nc),np.arange(nr))
    Mag_supp = np.where(M, Mag, 0)

    #weakEdgeMap=np.logical_and(Mag_supp>=low,Mag_supp<high)
    weakEdgeMap=Mag_supp>=low
    # initial EdgeMap with strong edges
    edgeMap=Mag_supp>=high
    # compute the edge direction from Ori
    edgeOri=Ori + np.pi/2
    right=x+np.cos(edgeOri)
    down=y+np.sin(edgeOri)
    left=x-np.cos(edgeOri)
    up=y-np.sin(edgeOri)
    # find neighbors in the edge direction
    
    while True:
      neighbor = interp2(Mag_supp, right, down)
      neighbor2 = interp2(Mag_supp, left, up)
      #should consider out of bound? yes
      e=0.0001
      bound=(right+e<0) | (right>nc-1+e) | (down+e<0) | (down>nr-1+e)
      op_bound=(left+e<0) | (left>nc-1+e) | (up+e<0) | (up>nr-1+e)
      neighbor[bound]=0
      neighbor2[op_bound]=0
      neigh_edge = neighbor>=high
      neigh2_edge = neighbor2>=high
      strong_neighbor=np.logical_or(neigh2_edge,neigh_edge)
      Mag_supp=np.where(np.logical_and(weakEdgeMap,strong_neighbor),np.maximum(neighbor,neighbor2),Mag_supp)
      temp_edgeMap=edgeMap+np.logical_and(weakEdgeMap,strong_neighbor)
      #update Mag_supp using logical_and(weakEdgeMap,strong_neighbor) -> update Mag supp with neighbor's value in order to get pass the threshold > high
      if np.array_equal(edgeMap,temp_edgeMap):
          break
      edgeMap=temp_edgeMap
      
          

    # try to link weak edges to strong edges until there is no change
   
    return edgeMap.astype(bool)

#optical flow functions
