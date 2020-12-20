"""
Team 21 CIS 581 Final Project

This is the main script to run. Give it a video, and it will output
an annotated video using the techniques explained in the report and the 
presentation.
"""

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

import numpy as np
import os, json, cv2, random
import math
from skimage import img_as_ubyte
from scipy.ndimage import center_of_mass
from sklearn.linear_model import LinearRegression
from optical_flow import * # our own file
import sys

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
import random
from helpers import drawPoints


import logging
logger = logging.getLogger(__name__)

# from detectron2.data.datasets import register_coco_instances

#%% Load the model:
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.MODEL.WEIGHTS ="model/outputfile.pth"# path to the model we trained #outputfile.pth is Jean's
predictor = DefaultPredictor(cfg)

#%% Functions

# function to draw an arrow on the frame
# tracks motion by the centers of mass of the masks
def drawArrow(id_list,prev_id_list,old_coords,prev_coords,old_bboxes,prev_bboxes,old_classes,prev_classes,vis,W,H):
    # a bit confusing: old is the current frame, prev is the previous frame
    
    for i in range(len(old_classes)):
        if old_classes[i]!= 3 and old_classes[i]!=9:
            continue
        # find the center of mass of the object's mask
        curr_mask = generateMaskWithCoordinates(old_coords[i],W,H)
        start = center_of_mass(curr_mask)
        start = start[::-1]
        if (np.isnan(start).any()):
            continue
        
        curr_id = id_list[i]
        
        # also find the bounding boxes to draw rectangles later
        curr_bb=old_bboxes.astype(float)[i]
        curr_bb=curr_bb.reshape(4,1)
        
        # find the center of mass of the same object in the previous frames
        # iterate through the previous frames
        prev_start = []
        for j in range(len(prev_id_list)):
            tmp_id_list = prev_id_list[j] # a frame's list of ids
            if (curr_id not in tmp_id_list):
                continue
            idx = list(tmp_id_list).index(curr_id) # find index of where the current id is inside this frame's idx
            tmp_class = prev_classes[j][idx]
            # doublecheck the class
            if (old_classes[i] != tmp_class):
                print('For some reason, the class isn\t the same')
            tmp_coords = prev_coords[j][idx]
            tmp_mask = generateMaskWithCoordinates(tmp_coords,W,H)
            tmp_start = center_of_mass(tmp_mask)
            tmp_start = tmp_start[::-1]
            if (np.isnan(tmp_start).any()):
                continue
            prev_start.append(tmp_start)
        
        # print(prev_start)
        if (len(prev_start) == 0):
            continue
        # fit a line to the points
        lm = LinearRegression(fit_intercept=False)
        coords = np.concatenate(([start],prev_start),axis=0) # array of the points
        y = coords[:,1]
        x = coords[:,0]
        y2 = y - y[0] # center the known point on the origin
        x2 = x - x[0]
        lm.fit(x2.reshape(-1,1),y2)
        slope = lm.coef_[0]
        if (slope==0):
            slope += 0.000001
        b = start[1]-(slope*start[0]) # find the actual intercept
        
        # find arrow properties
        # angle between their centers
        arrow_angle = np.arctan2(slope,1) # slope works here
        # magnitude of the change
        line_length = 30
        endpoint=(int(line_length*math.cos(arrow_angle))+start[0],int(line_length*math.sin(arrow_angle))+start[1])
        
        # find where the current trajectory intersects with the bottom of the image
        # the segment at the bottom where intersects are collisions
        minbound=0.3*(W/2)              # left area of boundary
        maxbound=(W/2)+(0.3*(W/2))      # right area of boundary

        intersect_loc_x=(H-b)/slope#(slope*line_y)+b
        cv2.rectangle(vis, (int(curr_bb[0]),int(curr_bb[1])), (int(curr_bb[2]),int(curr_bb[3])), (0,100,100), 3) 
        new_start = tuple([int(_) for _ in start])     # no good way to turn tuples of floats into tuples of integers for some reason
        new_end =   tuple([int(_) for _ in endpoint])
        if intersect_loc_x > minbound and intersect_loc_x < maxbound and slope > 0:
            cv2.arrowedLine(vis,new_start,new_end,(0,0,255),2)
        else:
            cv2.arrowedLine(vis,new_start,new_end,(0,255,0),2)
    return vis


# tracks and classifies objects
def classifyVideo(rawVideo):
    """

        Description: Generate and save tracking video
        Input:
        rawVideo: Raw video file name, String
        Instruction: Please feel free to use cv.selectROI() to manually select bounding box

    """

    cap = cv2.VideoCapture(rawVideo)
    imgs = []
    frame_cnt = 0
    
    out_folder = rawVideo[13:-4]+'/' # pulls the video's name out
    if not(os.path.exists('./output_videos/'+out_folder)):
        os.mkdir('./output_videos/'+out_folder)
    
    #random colors for 14 classes
    colors = [tuple(np.random.randint(256, size=3)) for _ in range(14)]
    # print(len(colors))
    # Initialize video writer for tracking video
    trackVideo = './output_videos/'+out_folder+'Output.mp4'
    path_trackVideo = './output_videos/'+out_folder+'Output_path.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #size = (int(cap.get(360)), int(cap.get(480)))
    writer = cv2.VideoWriter(trackVideo, fourcc, fps, size)
    path_writer = cv2.VideoWriter(path_trackVideo, fourcc, fps, size)

    #max number of features you will extract
    N=5

    #variables
    old_classes=np.array([])
    old_count=0
    old_masks=[]
    old_coords=np.array([])
    initNF=[]
    id_list=np.array([])
    count=0
    path_draw_dict = {}
    path_draw_color_dict = {}


    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret: break

        #writing video on vis
        path_draw_frame = frame.copy()
        vis = frame.copy() 
        #frames used for feature translation
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 
        frame_cnt += 1

        H,W = vis.shape[0],vis.shape[1]

        outputs = predictor(vis)
        v = Visualizer(vis[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.85)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #temp=outputs["instances"].to("cpu").pred_boxes.tensor.numpy().astype(float)
        #print("Check SHAPE",temp,temp[3])
        newimage=np.zeros((H,W))
        count=outputs["instances"].to("cpu").pred_classes.numpy().shape
        
        #put mask on frame using detectron2 visualizer
        for i in range(count[0]):
            class_num=outputs["instances"].pred_classes[i]
            if class_num != 3 and class_num!= 9:
                continue
            mask=outputs["instances"].to("cpu").pred_masks.numpy()[i].astype(int)*(i+1)
            #use current instance's class id to get corresponding color
            color=colors[class_num]
            for n in range(3):
                vis[:, :, n] = np.where(mask!= 0, (vis[:, :, n] * 0.5 + 0.5*color[n]),vis[:, :, n])
        
        cv2.imwrite('./output_videos/'+out_folder+'{}_1.jpg'.format(frame_cnt), img_as_ubyte(vis))
        

        if frame_cnt==1:
            #save first frame's features, bboxes, mask coordinates, count, and classes
            bboxes=outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
            masks=outputs["instances"].to("cpu").pred_masks.numpy()
            num=bboxes.shape[0]
            bboxes=bboxes.reshape(num,2,2)
            masks=masks.reshape(num,H,W)
            classes=outputs['instances'].to('cpu').pred_classes.numpy()
            old_masks, old_bboxes, old_classes, old_count, id_list = getCarAndHuman(masks,bboxes,classes,count[0],0,N,H,W)
            initNF,features=getFeatures(frame,bboxes,N)
            old_coords=generateAllCoordinates(old_masks,W,H)
            
            # history for trajectory prediction
            prev_classes = []
            prev_bboxes= []
            prev_classes = []
            prev_coords = []
            prev_id_list = []
            prev_count = []
        else:
            
            all_count=0
            all_bboxes,all_masks,all_features,all_coords,all_classes = np.array([]),np.array([]),np.array([]),np.array([]),[]
            all_featNum=[]
            all_id=np.array([])
    
            # get current NN prediction
            bboxes=outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
            num=bboxes.shape[0]
            bboxes=bboxes.reshape(num,2,2)
            masks=outputs["instances"].to("cpu").pred_masks.numpy()
            masks=masks.reshape(count[0],H,W)
            new_classes=outputs["instances"].to("cpu").pred_classes.numpy()
            masks, bboxes, new_classes, newcount, newid_list = getCarAndHuman(masks,bboxes,new_classes,count[0],np.amax(id_list),N,H,W)
            tmp_num,tmp_features=getFeatures(frame,bboxes,N)
            tmp_coords=generateAllCoordinates(masks,W,H)
            #current frame's class dictionary -> key: class, value: number of class
            unique, counts = np.unique(new_classes, return_counts=True)
            dictC=dict(zip(unique, counts)) 
            print("ALL classes",all_classes)
            print("old count",old_count," and count",newcount)
            print("old classes",old_classes)
            
            # look at all the previous predictions (NN and optical flow)
            # and determine if any optical flow needs to be applied to fix it
            for k in range(old_count):
                # print(count[0],"running:",k)
                cnt=0
                if old_classes[k]!= 9 and old_classes[k]!=3:
                    #we are generating masks only on human and cars
                    continue
                if old_classes[k] not in new_classes:
                    #class from previous frames are not detected in current frame -> optical flow to generate mask
                    # print("class",old_classes[k]," not found")
                    all_featNum,all_features,all_bboxes, coord,all_coords,all_classes, all_id, eraseObject=transformMask(initNF[k],frame,frame_old,all_featNum, all_features,all_bboxes,all_coords,all_classes,all_id,features[k],old_bboxes[k],old_coords[k],old_classes[k],id_list[k],H,W,N)
                    
                    if eraseObject==False:
                        tmp_coord=coord.reshape(H*W,2)
                        #generate mask on the frame with the transformed coordinates
                        mask=generateMaskWithCoordinates(tmp_coord,W,H)
                        color=colors[int(old_classes[k])]
                        for n in range(3):
                            vis[:, :, n] = np.where(mask!= 0, (vis[:, :, n] * 0.5 + 0.5*color[n]),vis[:, :, n])
                        all_count+=1
                    continue
                for j in range(newcount):
                    checkId=False
                    if new_classes[j] != 3 and new_classes[j] !=9:
                        continue
                    class_num=new_classes[j]
                    # print("DICT",dictC )
                    # print("Class Num:",class_num,type(class_num))
                    # print("get",dictC.get(class_num.item())," K",k)
                    if old_classes[k]==class_num:
                        old_mask=generateMaskWithCoordinates(old_coords[k],W,H)
                        intersect=np.logical_and(old_mask,masks[j])
                        interNum=np.count_nonzero(intersect)
                        maskNum=np.count_nonzero(old_mask)
                        newmaskNum=np.count_nonzero(masks[j])
                        # print("InterNum ",interNum," old maskNum ",maskNum, "new maskNum",newmaskNum)
                        if maskNum==0:
                            continue
                        # check for a mask with sizable overlap
                        if interNum>=0.7*maskNum:
                            # print("masks match")
                            # print("matching id",id_list[k])
                            #add to list
                            all_id=np.append(all_id,id_list[k])
                            all_classes.append(class_num)
                            tmp_bbox=bboxes[j].reshape(1,2,2)
                            all_bboxes=np.append(all_bboxes,tmp_bbox)
                            mask=masks[j]
                            mask=mask.reshape(1,H,W)
                            all_coords=np.append(all_coords,tmp_coords[j])
                            all_featNum.append(tmp_num[j])
                            all_features=np.append(all_features,tmp_features[j])
                            all_count+=1
                            # print("ADDED",all_featNum)
                            checkId=True
                            #fill zeros to the used mask so that it does not get used over again
                            masks[j]=np.zeros_like(masks[j])
                        else:
                            cnt+=1
                            if (dictC.get(class_num.item())>cnt):
                                # print("masks do not match but will have to look more : cnt",cnt)
                                pass
                            else:
                                print("masks do not match : optical flow to put mask on pic")
                                print("not matching id",id_list[k])
                                all_featNum,all_features,all_bboxes, coord,all_coords,all_classes, all_id, eraseObject=transformMask(initNF[k],frame,frame_old,all_featNum, all_features,all_bboxes,all_coords,all_classes,all_id,features[k],old_bboxes[k],old_coords[k],old_classes[k],id_list[k],H,W,N)
                                checkId=True
                                if eraseObject==False:
                                    tmp_coord=coord.reshape(H*W,2)
                                    mask=generateMaskWithCoordinates(tmp_coord,W,H)
                                    color=colors[int(old_classes[k])]
                                    for n in range(3):
                                        vis[:, :, n] = np.where(mask!= 0, (vis[:, :, n] * 0.5 + 0.5*color[n]),vis[:, :, n])
                                    all_count+=1
                        if checkId==True:
                            newid_list[j]=0
                            break
            
            print('Resulting newid_list',newid_list)
            for i in range(newid_list.size):
                if newid_list[i].astype(np.int8)!=0:
                    # print('Adding nonzero id to the list')
                    all_id=np.append(all_id,newid_list[i])
                    all_classes.append(new_classes[i])
                    tmp_bbox=bboxes[i].reshape(1,2,2)
                    all_bboxes=np.append(all_bboxes,tmp_bbox)
                    mask=masks[i]
                    mask=mask.reshape(1,H,W)
                    all_coords=np.append(all_coords,tmp_coords[i])
                    all_featNum.append(tmp_num[i])
                    all_features=np.append(all_features,tmp_features[i])
                    all_count+=1
            
            print("Count",all_count)
            print("########save frame",frame_cnt,"###########")
            
            # save the results of the previous frame needed for arrow drawing
            if (frame_cnt < 10): # just append if still early
                prev_count.append(old_count)
                prev_bboxes.append(old_bboxes.copy())
                prev_classes.append(old_classes.copy())
                prev_coords.append(old_coords.copy())
                prev_id_list.append(id_list.copy())
            else: # delete the oldest entry to save memory
                prev_count=prev_count[1:]
                prev_bboxes=prev_bboxes[1:]
                prev_classes=prev_classes[1:]
                prev_coords=prev_coords[1:]
                prev_id_list=prev_id_list[1:]
                
                prev_count.append(old_count)
                prev_bboxes.append(old_bboxes.copy())
                prev_classes.append(old_classes.copy())
                prev_coords.append(old_coords.copy())
                prev_id_list.append(id_list.copy())
                
            old_count = all_count
            old_coords = all_coords.reshape((old_count,H*W,2))
            old_bboxes = all_bboxes.reshape((old_count,2,2))
            initNF = all_featNum
            features=all_features
            id_list = all_id.copy()
            old_classes = all_classes.copy()
            
            # reshaping coordinates, bboxes, features
            old_coords=old_coords.reshape((old_count,H*W,2))
            old_bboxes=old_bboxes.reshape((old_count,2,2))
            features=features.reshape((old_count,N,2))
            
            
            #print("BBOXES",old_bboxes)
            # draw the arrows and bounding boxes
            drawArrow(id_list,prev_id_list,old_coords,prev_coords,old_bboxes,prev_bboxes,old_classes,prev_classes,vis,W,H)
            print("########save done for frame",frame_cnt,"###########")
        #print(features)
        print('ID List',id_list)

        ### path drawing starts ###
        
        list_index = 0

        ids_that_disappeared = []
        for key in path_draw_dict.keys():
          if key not in id_list:
            ids_that_disappeared.append(key)
        
        for id in ids_that_disappeared:
          del path_draw_dict[id]

        for box in old_bboxes: 
          mid_x = box[0][0] + (box[1][0] - box[0][0])/2
          mid_y = box[0][1] + (box[1][1] - box[0][1])/2
          curr_id = id_list[list_index]
          new_coords = [mid_x, mid_y]
          if curr_id in path_draw_dict:
            path_draw_dict[curr_id].append(new_coords)
          else: 
            path_draw_dict[curr_id] = [new_coords]
          list_index = list_index + 1

        for key in path_draw_dict.keys():
          for coords in path_draw_dict[key]:
            if key in path_draw_color_dict:
              color = path_draw_color_dict[key]
            else:
              ran1 = random.randrange(256)
              ran2 = random.randrange(256)
              ran3 = random.randrange(256)
              color = [ran1,ran2,ran3]
              path_draw_color_dict[key] = color

            drawPoints(path_draw_frame, coords[0], coords[1],color)
        ### path drawing ends ###
        
        #save frame   
        frame_old=frame.copy()
        # save to list
        imgs.append(img_as_ubyte(vis))
        
        # save image 
        #if (frame_cnt + 1) % 2 == 0:
        cv2.imwrite('./output_videos/'+out_folder+'{}_2.jpg'.format(frame_cnt), img_as_ubyte(vis))
        
        path_writer.write(path_draw_frame)
        cv2.imwrite('./output_videos/'+out_folder+'{}_trace.jpg'.format(frame_cnt), path_draw_frame)
        # Save video
        writer.write(vis)
        if (frame_cnt==100):
            break
        
    # Release video reader and video writer
    cap.release()
    writer.release()
    
    return

#%% Main function definition
if __name__=='__main__':

    # reminder to include the video to look at
    if len(sys.argv) < 2:
        print('usage: python main.py <video>')
        sys.exit()
    
    videoIn = sys.argv[1]
    video = 'input_videos/'+videoIn
    
    if not(os.path.exists('./output_videos/')):
        os.mkdir('./output_videos/')
    
    classifyVideo(video)
