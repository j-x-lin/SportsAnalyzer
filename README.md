# Sports Player Tracker #

This is a project to take in sports film footage and track players' movements throughout the video, to for example see the different routes that receivers run on any given play, or see how the defense reacts to the snap by shifting coverage.

You can run this for yourself by cloning this repo, then running `python pipeline.py` in the resulting folder. The movement traces will appear in `movements.jpg`. 

# Data #

I created a dataset by taking All-22 film footage from NFL games at both college and professional levels. For each video, I only kept the portions where you can see all 22 players on the field, and split each video up into frames.

# Approach #
Initially, I tried to run simple edge detection to see if I could come up with clear outlines of the players in order to run object detection. However, there were a number of issues with this approach. The first main issue was that basic edge detection would often wrongfully detect "noise" objects such as number markings on the field, TV drones, etc., which we do not want to keep track of as the play progresses. The other, more significant issue was that unlike TV footage, All-22 film footage comes from a camera high above the field such that it can capture all 22 players in most frames. As a result, the footage is not very "clear", so there is a lot of noise which interferes with edge detection.

As a result, I decided to use YOLOv5 (https://github.com/ultralytics/yolov5), which is pre-trained on the COCO dataset, to attempt to locate players and label bounding boxes around each player.

There was one (large) remaining problem to solve though: how to combine different frames? Since the camera is tracking all 22 players, it is moving between different frames, so I needed to create a reference coordinate system for all the frames so that I could plot points in the same coordinate system. A further complication is that while panorama stitching is very effective on large landscapes, it is less effective at stitching together consecutive frames of a video when the players and camera are both moving between frames. Hence, instead of repeatedly stitching together images I only saved the homography matrices to refer all images to the first frame's coordinate system. 

# Demo Video #
https://youtu.be/7v0H1B1-O8s

# Future Directions #
Here are some possible future directions I hope to take this project:

1. Finetune the YOLOv5 network to distinguish between players and referees

2. Improve the recognition by using a different color for different players, and keep track of different players between consecutive frames

3. Improve the homography calculation even further 
