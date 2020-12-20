# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:29:32 2020

@author: Nicholas
"""

from PIL import Image 					# (pip install Pillow)
import numpy as np                                 	# (pip install numpy)
from skimage import measure                        	# (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon 	# (pip install Shapely)
import os
import json

#from create_annotations import *

# Define which colors match which categories in the images
category_ids = {
    '(0, 0, 0)': 0, # Nothing
    '(107, 142, 35)': 1, # Trees
    '(128, 64, 128)': 2, # Road
    '(220, 20, 60)': 3, # People
    '(70, 130, 180)': 4, # Sky
    '(70, 70, 70)': 5, # Building
    '(152, 251, 152)': 6, # Grass
    '(220, 220, 0)': 7, # Sign
    '(153, 153, 153)': 8, # Pole
    '(0, 0, 142)': 9, # Car
    '(244, 35, 232)': 10, # Sidewalk
    '(250, 170, 30)': 11, # StopLight
    '(0, 0, 70)': 12, # Vehicle
    '(190, 153, 153)': 13, # Wall

}


def create_sub_masks(mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    polygons = []
    segmentations = []
    j = 0
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        
        if(poly.is_empty):
            # Go to next iteration, dont save empty values in list
            continue

        polygons.append(poly)
        
        try:
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)
        except:
            continue
      
            
    
    return polygons, segmentations

def create_image_annotation(file_name, width, height, image_id):
    images = {
        'file_name': file_name,
        'height': height,
        'width': width,
        'id': image_id
    }

    return images

# Helper function to get absolute paths of all files in a directory
def absolute_file_paths(directory):
    mask_images = []

    for root, dirs, files in os.walk(os.path.abspath(directory)):
        for file in files:
            # Filter only for images in folder         
            if '.png' or '.jpg' in file: 
                mask_images.append(os.path.join(root, file))
    return mask_images

def create_annotation_format(polygon, segmentation, image_id, category_id, annotation_id):
    
    try:
        min_x, min_y, max_x, max_y = polygon.bounds
        width = max_x - min_x
        height = max_y - min_y
        bbox = (min_x, min_y, width, height)
        area = polygon.area
    except:  
        min_x, min_y, max_x, max_y = 15,15,35,35
        width = max_x - min_x
        height = max_y - min_y
        bbox = (min_x, min_y, width, height)
        area = polygon.area

    annotation = {
        'segmentation': segmentation,
        'area': area,
        'iscrowd': 0,
         'image_id': image_id,
        'bbox': bbox,
        'category_id': category_id,
        'id': annotation_id
    }

    return annotation

# Create the annotations of the ECP dataset (Coco format) 
coco_format = {
    "images": [
        {
        }
    ],
    "categories": [
        {
            "supercategory": "trees",
            "id": 1,
            "name": 'trees'
        },
        {
            "supercategory": "road",
            "id": 2,
            "name": 'road'
        },
        {
            "supercategory": "human",
            "id": 3,
            "name": 'human'
        },
        {
            "supercategory": "sky",
            "id": 4,
            "name": 'sky'
        },
        {
            "supercategory": "building",
            "id": 5,
            "name": 'building'
        },
        {
            "supercategory": "grass",
            "id": 6,
            "name": 'grass'
        },
        {
            "supercategory": "sign",
            "id": 7,
            "name": 'sign'
        },
        {
            "supercategory": "pole",
            "id": 8,
            "name": 'pole'
        },
                {
            "supercategory": "car",
            "id": 9,
            "name": 'car'
        },
        {
            "supercategory": "sidewalk",
            "id": 10,
            "name": 'sidewalk'
        },
        {
            "supercategory": "stop light",
            "id": 11,
            "name": 'stop light'
        },
        {
            "supercategory": "vehicle",
            "id": 12,
            "name": 'vehicle'
        },
        {
            "supercategory": "wall",
            "id": 13,
            "name": 'wall'
        },
        {
            "supercategory": "outlier",
            "id": 0,
            "name": 'outlier'
        },
        
    ],
    "annotations": [
        {
        }
    ]
}


# Get 'images' and 'annotations' info 
def images_annotations_info(maskpath):
    # This id will be automatically increased as we go
    annotation_id = 1

    annotations = []
    images = []
    
    # Get absolute paths of all files in a directory
    mask_images = absolute_file_paths(maskpath)
    #print("mask_images=",mask_images)
    print("Total=",len(mask_images))
    count=0
    
    for image_id, mask_image in enumerate(mask_images, 1):
        file_name = os.path.basename(mask_image).split('.')[0] + ".jpg"
        
        print("Images Remaining=",len(mask_images)-count)
        count+=1
        
        if count>=1000:
            return images, annotations
            
        
        # image shape
        mask_image_open = Image.open(mask_image)
        w, h = mask_image_open.size
        #print("w=",w)
        # 'images' info 
        image = create_image_annotation(file_name, w, h, image_id)
        images.append(image)

        sub_masks = create_sub_masks(mask_image_open, w, h)
        for color, sub_mask in sub_masks.items():
            #return category_ids,color
            #pass
            #print(category_ids)
            #print(color)
            if color in category_ids:
                category_id = category_ids[color]
    
                # 'annotations' info
                polygons, segmentations = create_sub_mask_annotation(sub_mask)
    
                # Three labels are multipolygons in our case: wall, roof and sky
#                if(category_id == 1 or category_id == 2 or category_id == 4 or category_id == 6 or category_id == 10 or category_id == 13):
#                    # Combine the polygons to calculate the bounding box and area
#                    
#                    try:
#                        multi_poly = MultiPolygon(polygons)
#                                        
#                        annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)
#        
#                        annotations.append(annotation)
#                        annotation_id += 1
#                    except:
#                        
#                        for i in range(len(polygons)):
#                            # Cleaner to recalculate this variable
#                            segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
#                            
#                            annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
#                            
#                            annotations.append(annotation)
#                            annotation_id += 1
                        
#                else:
                for i in range(len(polygons)):
                    # Cleaner to recalculate this variable
                    segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                    
                    annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                    
                    annotations.append(annotation)
                    annotation_id += 1
        
    return images, annotations

if __name__ == '__main__':
    for keyword in ['train', 'val']:
        mask_path =r'bdd_dataset\bdd100k\seg\color_labels\train'# 'dataset/{}_mask'.format(keyword)
        #test,color=images_annotations_info(mask_path)
        coco_format['images'], coco_format['annotations'] = images_annotations_info(mask_path)
        #print(json.dumps(coco_format))
        with open(r'bdd_dataset\labels_coco\redo\training_1000v2.json'.format(keyword),'w') as outfile:
            json.dump(coco_format, outfile)
            
            
        f = open(r'bdd_dataset\labels_coco\redo\training_1000v2.json','r')
        a = ['_train_color.']
        lst = []
        for line in f:
            for word in a:
                if word in line:
                    line = line.replace(word,'.')
            lst.append(line)
        f.close()
        f = open(r'bdd_dataset\labels_coco\redo\training_corrected_1000v2.json','w')
        for line in lst:
            f.write(line)
        f.close()    
         
        break
            
            
            
            