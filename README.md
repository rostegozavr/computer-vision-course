# computer-vision-course

![screenshot](https://github.com/rostegozavr/computer-vision-course/blob/master/images/screenshot.png)

## Datasets and trained nets

Trained Yolo_v3 can be found [here](https://pjreddie.com/darknet/yolo/). 

Also I use pictures from [MIO-TCD dataset](http://podoce.dinf.usherbrooke.ca/challenge/dataset/) to train my net for transport classification (just **car**, **truck**, **bus** for a start).

## Architecture of classification model

![netron](https://github.com/rostegozavr/computer-vision-course/blob/master/images/netron.png)

## Training process
1. Over 9000 images from MIO-TCD were resized to (96,64)
2. Images were separated for training and test
3. Trained 100 epochs with batch_size = 32
4. Detecting objects in video, transport classification

![process](https://github.com/rostegozavr/computer-vision-course/blob/master/images/charts.png)