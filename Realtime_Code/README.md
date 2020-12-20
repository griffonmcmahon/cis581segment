# This folder contains code for the two real time implementations created during this project.
#### This includes code for:
* Real Time Homography Using Object Detection
* Real Time Object Detection

#### Real Time Homography
The file Code_Simple_Homography.ipynb uses the object detection system implemented in this project to warp videos from street level view, to top down view.

In order to run this file, move it into Google Drive and open it with Colab.  From there, specify the location of the outputfil.pth file on your google drive in the code(box 6 in the ipynb notebook).  If you have 
not already downloaded it, you can find it here: https://drive.google.com/u/1/uc?id=1mXQLFDdJbieIQLGIqeju9QdufQpX13j9&export=download 

Next, set your input video location.  It is recommended you use the file twolane_example.mp4 located inside this folder.

Lastly, specify the output location, where you would like the output video to be saved.

Then run the notebook(processing time for this video should be around 41 seconds)

If the last section of code gives an error after running, try running again as it should work on the second run.

#### Real Time Object Detection
To run the real time object detection, download Realtime_Image_Segmentation.ipynb and upload it to google drive.  Then open it using Colab.
From there, specify the location of the outputfil.pth file on your google drive in the code(box 6 in the ipynb notebook).  If you have 
not already downloaded it, you can find it here: https://drive.google.com/u/1/uc?id=1mXQLFDdJbieIQLGIqeju9QdufQpX13j9&export=download 

Next set your input video location.  The .mp4 file named 320_240_vid.mp4 in this folder should work well.

Lastly, specify the output location, where you would like the output video to be saved.

Then run the notebook(processing time for this video should be around 41 seconds)

If the last section of code gives an error after running, try running again as it should work on the second run.
