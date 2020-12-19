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
from optical_flow import * # our own file
import sys

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog


import logging
logger = logging.getLogger(__name__)

# from detectron2.data.datasets import register_coco_instances

#%% Load the model:
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.MODEL.WEIGHTS ="model/Highres.pth"# path to the model we trained
predictor = DefaultPredictor(cfg)

#%% Functions

# function to draw an arrow on the frame
# tracks motion by the centers of mass of the masks
def drawArrow(obj_idx,curr_idx,old_coords,prev_coords,old_bboxes,prev_bboxes,old_classes,prev_classes,vis,W,H):
    # a bit confusing: old is the current frame, prev is the previous frame
    minbound=0.3*(W/2)
    maxbound=(W/2)+(0.3*(W/2))
    
    for i in curr_idx:
        if old_classes[i]!= 3 and old_classes[i]!=9:
            continue
        if (old_classes[i] != prev_classes[obj_idx[i]]):
            print('For some reason, the class isnt the same')
            print(old_classes)
            print(prev_classes)
            print(obj_idx)
        curr_mask = generateMaskWithCoordinates(old_coords[i],W,H)
        curr_bb=old_bboxes.astype(float)[i]
        start = center_of_mass(curr_mask)
        start = start[::-1]
        prev_mask = generateMaskWithCoordinates(prev_coords[obj_idx[i]],W,H)
        prev_bb=prev_bboxes.astype(float)[obj_idx[i]]
        prev_start = center_of_mass(prev_mask)
        prev_start = prev_start[::-1]
        if (np.isnan(start).any()) or (np.isnan(prev_start).any().any()):
            continue
        # sum=np.sum(curr_mid)
        # prev_sum = np.sum(prev_mid)
        # if np.isnan(sum) or np.isnan(prev_sum):
        #     continue
        curr_bb=curr_bb.reshape(4,1)
        prev_bb=prev_bb.reshape(4,1)
        # find box's midpoints
        # start=(int((curr_mid[2]+curr_mid[0])/2),int((curr_mid[3]+curr_mid[1])/2))
        # prev_start=(int((prev_mid[2]+prev_mid[0])/2),int((prev_mid[3]+prev_mid[1])/2))
        # angle between their centers
        arrow_angle = np.arctan2(start[1]-prev_start[1],start[0]-prev_start[0])
        # magnitude of the change
        mag = np.linalg.norm([start[1]-prev_start[1],start[0]-prev_start[0]])
        line_length = 30#mag*6
        endpoint=(int(line_length*math.cos(arrow_angle))+start[0],int(line_length*math.sin(arrow_angle))+start[1])
        # endpoint = (prev_start[0],prev_start[1])
        x_curr=start[0]
        y_curr=start[1]
        x_prev=prev_start[0]
        y_prev=prev_start[1]
        if (x_curr-x_prev==0):
            x_curr+=0.0000001
        if (y_curr-y_prev==0):
            y_curr+=0.0000001
        slope=((y_curr-y_prev)/(x_curr-x_prev))
        b=y_curr-(slope*x_curr)
        intersect_loc_x=(H-b)/slope#(slope*line_y)+b
        cv2.rectangle(vis, (int(curr_bb[0]),int(curr_bb[1])), (int(curr_bb[2]),int(curr_bb[3])), (0,100,100), 3) 
        new_start = tuple([int(_) for _ in start])
        new_end =   tuple([int(_) for _ in endpoint])
        if intersect_loc_x > minbound and intersect_loc_x < maxbound and y_prev<y_curr:
            cv2.arrowedLine(vis,new_start,new_end,(0,0,255),2)
        else:
            cv2.arrowedLine(vis,new_start,new_end,(0,255,0),2)
    return vis

def object_track(old_classes,prev_classes,old_coords,prev_coords,W,H):
    # outputs indexing for the previous frame's matches to the current frame's
    # usage: curr_thing[curr_idx[i]] is the same object as prev_thing[prev_idx[curr_idx[i]]]
    idx = np.zeros_like(old_classes) + 99999
    curr_idx = []
    # look at the current object...
    for i in range(len(old_classes)):
        curr_mask = generateMaskWithCoordinates(old_coords[i],W,H)
        curr_com = np.array(center_of_mass(curr_mask))
        best_dist = 99999
        # ...and try to match them to all of the old ones
        for j in range(len(prev_classes)):
            # but make sure they're the same class first
            if (old_classes[i] != prev_classes[j]):
                continue
            prev_mask = np.array(generateMaskWithCoordinates(prev_coords[j],W,H))
            #intersect = np.sum(np.logical_and(curr_mask,prev_mask))
            prev_com = np.array(center_of_mass(prev_mask))
            dist = np.linalg.norm(curr_com-prev_com)
            # see if it overlaps better than another one
            if (dist < best_dist):
                best_dist = dist
                idx[i] = j
        # make sure an actual match was found
        if (best_dist != 99999): # the bestIntersect isn't zero anymore
            curr_idx.append(i)
            
    return idx,curr_idx

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
    trackVideo = './output_videos/'+out_folder+'Output2.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #size = (int(cap.get(360)), int(cap.get(480)))
    writer = cv2.VideoWriter(trackVideo, fourcc, fps, size)

    #max number of features you will extract
    N=5

    #Lucas Kanade param #TODO: look into deleting this
    lk_params = dict( winSize  = (11,11),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    #variables
    old_classes=np.array([])
    old_count=0
    old_masks=[]
    old_coords=np.array([])
    initNF=[]


    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret: break
        #rotate the video frame 
        #
        #frame=np.rot90(frame)
        #writing video on vis
        vis = frame.copy() 
        #frames used for feature translation
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 
        frame_cnt += 1
        #if frame_cnt<230:
          #continue
        H,W = vis.shape[0],vis.shape[1]
        #minimum bound and maximum bound for arrow
        minbound=0.3*(W/2)
        maxbound=(W/2)+(0.3*(W/2))
        outputs = predictor(vis)
        v = Visualizer(vis[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.85)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #temp=outputs["instances"].to("cpu").pred_boxes.tensor.numpy().astype(float)
        #print("Check SHAPE",temp,temp[3])
        newimage=np.zeros((H,W))
        count=outputs["instances"].to("cpu").pred_classes.numpy().shape
        
        #put mask on frame using detectron2 visualizer
        for i in range(count[0]):
            mask=outputs["instances"].to("cpu").pred_masks.numpy()[i].astype(int)*(i+1)
            #use current instance's class id to get corresponding color
            class_num=outputs["instances"].pred_classes[i]
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
            initNF,features=getFeatures(frame,bboxes,N)
            old_classes=outputs["instances"].to("cpu").pred_classes.numpy()
            old_bboxes=bboxes
            old_masks=masks
            old_coords=generateAllCoordinates(old_masks,W,H)
            old_count=count[0]
        else:
            all_count=0
            all_bboxes,all_masks,all_features,all_masks,all_coords,all_classes = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),[]
            all_featNum=[]
            """
            #keep tracking objects that were detected in the first frame
            for k in range(old_count):
              print("FOR")
              all_featNum,all_features,all_bboxes, coord,all_coords,all_classes, eraseObject=transformMask(initNF[k],frame,frame_old,all_featNum, all_features,all_bboxes,all_coords,all_classes,features[k],old_bboxes[k],old_coords[k],old_classes[k],H,W,N)
            features=all_features
            old_bboxes=all_bboxes
            old_coords=all_coords
            old_classes=all_classes
            """
            #current frame's class dictionary -> key: class, value: number of class
            new_classes=outputs["instances"].to("cpu").pred_classes.numpy()
            unique, counts = np.unique(new_classes, return_counts=True)
            dictC=dict(zip(unique, counts))
            # print("ALL classes",all_classes)
            print("old count",old_count," and count",count)
            print("old classes",old_classes)
            # try to iterate through all the previously identified objects
            for k in range(old_count):
                print(count[0],"running:",k)
                cnt=0
                #we are generating masks only on human and cars
                if old_classes[k]!= 9 and old_classes[k]!=3:
                    continue
                if old_classes[k] not in new_classes:
                    #class from previous frames are not detected in current frame -> optical flow to generate mask
                    print("class",old_classes[k]," not found")
                    all_featNum,all_features,all_bboxes, coord,all_coords,all_classes, eraseObject=transformMask(initNF[k],frame,frame_old,all_featNum, all_features,all_bboxes,all_coords,all_classes,features[k],old_bboxes[k],old_coords[k],old_classes[k],H,W,N)
                    
                    #all_featNum,all_features,all_bboxes, coord,all_coords,all_classes, eraseObject=transformMask(initNF[k],frame,frame_old,all_featNum, all_features,all_bboxes,all_coords,all_classes,features[k],old_bboxes[k],old_coords[k],old_classes[k],H,W,N,lk_params)
                    
                    if eraseObject==False:
                        tmp_coord=coord.reshape(H*W,2)
                        #generate mask on the frame with the transformed coordinates
                        mask=generateMaskWithCoordinates(tmp_coord,W,H)
                        color=colors[int(old_classes[k])]
                        for n in range(3):
                            vis[:, :, n] = np.where(mask!= 0, (vis[:, :, n] * 0.5 + 0.5*color[n]),vis[:, :, n])
                        all_count+=1
                    continue
                for j in range(count[0]):
                    class_num=new_classes[j]
                    mask=outputs["instances"].to("cpu").pred_masks.numpy()[j].astype(int)*(j+1)
                    #print("DICT",dictC )
                    #print("Class Num:",class_num,type(class_num))
                    #print("get",dictC.get(class_num.item())," K",k)
                    if old_classes[k]==class_num:
                        old_mask=generateMaskWithCoordinates(old_coords[k],W,H)
                        intersect=np.logical_and(old_mask,mask)
                        interNum=np.count_nonzero(intersect)
                        maskNum=np.count_nonzero(old_mask)
                        #print("InterNum ",interNum," maskNum ",maskNum)
                        if interNum>=0.5*maskNum:
                            print("masks overlap")
                            break
                        else:
                            cnt+=1
                            if (dictC.get(class_num.item())>cnt):
                                #print("masks do not match but will have to look more : cnt",cnt)
                                pass
                            else:
                                print("no masks match : use optical flow to put mask on frame")
                                all_featNum,all_features,all_bboxes, coord,all_coords,all_classes, eraseObject=transformMask(initNF[k],frame,frame_old,all_featNum, all_features,all_bboxes,all_coords,all_classes,features[k],old_bboxes[k],old_coords[k],old_classes[k],H,W,N)
                                
                                #all_featNum,all_features,all_bboxes, coord,all_coords,all_classes, eraseObject=transformMask(initNF[k],frame,frame_old,all_featNum, all_features,all_bboxes,all_coords,all_classes,features[k],old_bboxes[k],old_coords[k],old_classes[k],H,W,N,lk_params)
                                if eraseObject==False:
                                    tmp_coord=coord.reshape(H*W,2)
                                    mask=generateMaskWithCoordinates(tmp_coord,W,H)
                                    color=colors[int(old_classes[k])]
                                    for n in range(3):
                                        vis[:, :, n] = np.where(mask!= 0, (vis[:, :, n] * 0.5 + 0.5*color[n]),vis[:, :, n])
                                    all_count+=1
            print("Count",all_count)
            print("########save frame",frame_cnt,"###########")
            bboxes=outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
            num=bboxes.shape[0]
            bboxes=bboxes.reshape(num,2,2)
            masks=outputs["instances"].to("cpu").pred_masks.numpy()
            # save the results of the previous frame needed for arrow drawing
            prev_count = old_count
            prev_bboxes= old_bboxes.copy()
            prev_classes = old_classes.copy()
            prev_coords = old_coords.copy()
            
            if all_count>0:
                #optical flow was used at least once, save feature points, classes, mask coordinates, bboxes from previous frames and current frame
                old_count=all_count+count[0]
                old_classes=np.append(np.array(all_classes),new_classes)
                tmp_coords=generateAllCoordinates(masks,W,H)
                old_coords=np.append(all_coords,tmp_coords)
                old_bboxes=np.append(all_bboxes,bboxes)
                old_coords=old_coords.reshape((old_count,H*W,2))
                #masks=generateAllMasksWithCoordinates(old_coords,W,H)
                numF,features=getFeatures(frame,bboxes,N)
                initNF=np.append(all_featNum,numF)
                features=np.append(all_features,features)
            elif all_count==0:
                #no optical flow was used, save only current frame's feature points, bboxes, masks, count, and classes
                old_count=count[0]
                tmp_coords=generateAllCoordinates(masks,W,H)
                old_coords=np.append(all_coords,tmp_coords)
                old_coords=old_coords.reshape((old_count,H*W,2))
                #masks=generateAllMasksWithCoordinates(old_coords,W,H)
                numF,features=getFeatures(frame,bboxes,N)
                old_classes=new_classes
                initNF=numF
                old_bboxes=bboxes
        
            # reshaping coordinates, bboxes, features
            old_coords=old_coords.reshape((old_count,H*W,2))
            old_bboxes=old_bboxes.reshape((old_count,2,2))
            features=features.reshape((old_count,N,2))
            
            # determine which objects in which classes array are the same
            #TODO: write this function
            obj_idx,curr_idx = object_track(old_classes,prev_classes,old_coords,prev_coords,W,H)
            
            #print("BBOXES",old_bboxes)
            drawArrow(obj_idx,curr_idx,old_coords,prev_coords,old_bboxes,prev_bboxes,old_classes,prev_classes,vis,W,H)
            #drawArrow(obj_idx,curr_idx,old_coords,prev_coords,old_bboxes,prev_bboxes,old_classes,prev_classes,vis,W,H)
        #print(features)
        # # display the bbox
        #for f in range(old_count):
        #cv2.rectangle(vis, tuple(old_bboxes[0,0].astype(np.int32)), tuple(old_bboxes[0,1].astype(np.int32)), (0,0,255), thickness=2)
        """
        # # display feature points
        for f in range(old_count):
          #if old_classes[f]!=9:
          #  continue
          new_FList = features[f]
          for feat in new_FList:
            cv2.circle(vis, tuple(feat.astype(np.int32)), 2, (0,0,255), thickness=-1)
        """
        
        #save frame   
        frame_old=frame.copy()
        # save to list
        imgs.append(img_as_ubyte(vis))
        
        # save image 
        #if (frame_cnt + 1) % 2 == 0:
        cv2.imwrite('./output_videos/'+out_folder+'{}_2.jpg'.format(frame_cnt), img_as_ubyte(vis))
        
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