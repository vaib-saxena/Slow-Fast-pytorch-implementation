# Slow-Fast-pytorch-implementation with Colab notebook
![](data/pam.gif)

This is just a cleaned up and workable verison of the repo shared here https://github.com/MagicChuyi/SlowFast-Network-pytorch that was missing weights and had few bugs. I intend to make some changes as I proceed. 

As mentioned in the original repo, this gives a real time activity detection by detecting using YOLOv3, tracking using DeepSORT and activity detection using Slow Fast.


# Run the demo on your own data

1.Clone the repository: git clone https://github.com/vaib-saxena/Slow-Fast-pytorch-implementation.git

2.Download Yolo v3 weights: https://drive.google.com/file/d/1SSpVueL6W_4BE3sFDkzAgdMd35Mtl2N5/view?usp=sharing and paste in the directory

3.Download DeepSort re-id weights: https://drive.google.com/file/d/1bwLHXS5TocUfDL2-iLNJLs8WfUOZtg9B/view?usp=sharing and paste in deep\checkpoint directory

4.Download Pre-trained SlowFast Network weights: https://drive.google.com/file/d/1ooE-qh7LBL7kWceZRHPyIIBslWCBwdwy/view?usp=sharing and paste in the directory

5.Modify the weights path and your video path in video_demo.py.

6.Run video_demo.py.

# Colab notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vaib-saxena/Slow-Fast-pytorch-implementation/blob/master/slow_fast.ipynb)


# Dependencies
- python 3 (python2 not sure)
- numpy
- scipy
- opencv-python
- torch >= 1.0.0
- torchvision = 0.2.1
- youtube-dl
- ffmpeg


# Reference
- paper: [Slow Fast Networks](https://arxiv.org/pdf/1812.03982.pdf)

- code: [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast)

- https://github.com/MagicChuyi/SlowFast-Network-pytorch

- paper: [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

- code: [nwojke/deep_sort](https://github.com/nwojke/deep_sort)

- paper: [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

- code: [Joseph Redmon/yolov3](https://pjreddie.com/darknet/yolo/)
