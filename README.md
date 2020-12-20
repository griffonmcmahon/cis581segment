# CIS 581 Final Project: Team 21
### Nicholas Marion, Griffon McMahon, Jean Park, Meiwen Zhou

This program will apply image segmentation enhanced with optical flow to the first 100 frames of an input video.

You can find the GitHub repository for our code at [this link](https://github.com/griffonmcmahon/cis581segment).

### Installation
Having this folder on your computer is almost enough, but the model used for segmentation must be downloaded from [this Google Drive link](https://drive.google.com/u/1/uc?id=1mXQLFDdJbieIQLGIqeju9QdufQpX13j9&export=download).
Click the Download icon in the top right. This should let you download "outputfile.pth".
Place Highres.pth in the "./model/" folder.

Also, ensure you have the Python packages listed in requirements.txt. Note that, as this project uses Detectron2, **running on Linux or OS X is recommended**.

### Usage
1. Place an input video into the "input_videos" directory.
2. Open a terminal in the "cis581segment" directory.
3. Run  

		$ python main.py <video>  

	For example:  

		$ python main.py Easy.mp4  
	Currently, we only support using .mp4's as inputs.
4. The output frames and videos can be found in the "optresults" directory.

### Folder Structure
* ./Additional_Code_Created/: Contains additional files used to allow neural network training.
* ./Realtime_Code/: Contains jupyter notebooks for real-time classification and homography. Instructions for installing and running them are located in the folder's own README.
* ./input_videos/: Contains .mp4 files to run the script with. We recommend "Easy.mp4".
* ./model/: Contains the neural network model used to make predictions. Must be downloaded as noted in "Installation".
* ./output_videos/: Contains a folder for each video given to the script. 
	For example, the results from running the script on Easy.mp4 will be stored in "./output_videos/Easy/". **All result videos are stored as .mp4 files in these folders.**
* helpers.py: Contains helper functions for optical flow.
* main.py: The main part of the script. Run this to use the script.
* optical_flow.py: Contains most of the optical flow code converted from Project 4.5.
* requirements.txt: Contains a list of Python packages that must be installed.

### Real-Time Code
To test our code for real-time object detection and homography, see the readme inside the Realtime_Code folder.
### Unexpected Issues?
Please email one of us if there are package issues or crashes.