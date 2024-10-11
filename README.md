# Sports Player Tracker #

This is a project to take in sports film footage and track players' movements throughout the video, to for example see the different routes that receivers run on any given play, or see how the defense reacts to the snap by shifting coverage.

# How to Run #
After cloning this repo, upload any video footage **as an MP4 file** to the `/data/videos` folder, and add the video name to the `/data/videos/index` file (e.g. if your video is `play.mp4` then add in "play" on a new line in the index file). To run the whole workflow, first run `make` to compile the c library, then run `python main.py` in the project directory.

# Data #
These models were trained on a sample of all-22 footage from NFL and college games.

# Approach #
Initially, I tried to run simple edge detection to see if I could come up with clear outlines of the players in order to run object detection. However, there were a number of issues with this approach. The first main issue was that basic edge detection would often wrongfully detect "noise" objects such as number markings on the field, TV drones, etc., which we do not want to keep track of as the play progresses. The other, more significant issue was that unlike TV footage, All-22 film footage comes from a camera high above the field such that it can capture all 22 players in most frames. As a result, the footage is not very "clear", so there is a lot of noise which interferes with edge detection.

As a result, I decided to use YOLOv11 (https://docs.ultralytics.com/models/yolo11/), which is pre-trained on the COCO dataset, to attempt to locate players and label bounding boxes around each player.

There was one (large) remaining problem to solve though: how to combine different frames? Since the camera is tracking all 22 players, it is moving between different frames, so I needed to create a reference coordinate system for all the frames so that I could plot points in the same coordinate system. A further complication is that while panorama stitching is very effective on large landscapes, it is less effective at stitching together consecutive frames of a video when the players and camera are both moving between frames. Hence, instead of repeatedly stitching together images I only saved the homography matrices to refer all images to the first frame's coordinate system. 

# Demo Video #
https://youtu.be/7v0H1B1-O8s

# Future Directions #
Here are some possible future directions I hope to take this project:

1. Finetune the YOLOv11 network to distinguish between players and referees or train my own object detection model to do this task.

2. Improve the recognition by using a different color for different players, and keep track of different players between consecutive frames

3. Improve the organization of information in the movements graph, such as coloring all presnap motion the same color range, then using different colors for postsnap motion.