from __future__ import division
import time
import torch
import copy
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random
import pickle as pkl
import argparse
from deep_sort import DeepSort
from collections import deque
from backbone.base import Base as BackboneBase
from config.train_config import TrainConfig
from config.eval_config import EvalConfig
from config.config import Config
from model import Model
import csv
import pandas as pd

def index2class():
    file_path = '/content/Slow-Fast-pytorch-implementation/ava/ava_action_list_v2.0.csv'
    with open(file_path) as f:
        i2c_dic = {line.split(',')[0]: line.split(',')[1] for line in f}
    return i2c_dic


def get_test_input(input_dim, CUDA):
    img = cv2.imread("/content/Slow-Fast-pytorch-implementation/imgs/dog.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def normalize(frame):
    # Normalize the buffer
    # buffer = (buffer - 128)/128.0
    frame = (frame - np.array([[[128.0, 128.0, 128.0]]]))/128.0
    return frame

def to_tensor(buffer):
    # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
    # D = Depth (in this case, time), H = Height, W = Width, C = Channels
    return buffer.transpose([3, 0, 1, 2])

para = []
def imshow(bboxes, labels, probs,ids,count):


    for bbox, lables, prob,i in zip(bboxes, labels, probs,ids):
      #with open('activity1.csv', 'w', newline='') as file:
        #writer = csv.writer(file)
        #writer.writerow([round(p, 2), index2class()[str(lable)].split("(")[0], i])

        count_2 = 0
        for lable, p in zip(lables, prob):
            count_2 = count_2 + 1
            bbox = np.array(bbox)
            lable = int(lable)
            p = float(p)
            rat = (ZZ,i, round(p, 2),index2class()[str(lable)].split("(")[0])
            para.append(rat)
            #print(para)
            #print(index2class()[str(lable)].split("(")[0])
            #print(i)
            #print(frame)
            real_x_min = int(bbox[0])
            real_y_min = int(bbox[1])
            real_x_max = int(bbox[2])
            real_y_max = int(bbox[3])
            # Draw rectangle on each frame, frame frame, (four coordinate parameters), (color), width
            cv2.rectangle(frame, (real_x_min, real_y_min), (real_x_max, real_y_max), (0, 255, 255),
                          2)  # yellow color
            cv2.putText(frame, index2class()[str(lable)].split("(")[0] + ':' + str(round(p, 2)),
                        (real_x_min + 15, real_y_max - 15 * count_2),
                        cv2.FONT_HERSHEY_COMPLEX, \
                        0.5, (0, 255, 255), 1, False)
            cv2.putText(frame, "person:"+str(i),
                        (real_x_min + 10, real_y_min + 20),
                        cv2.FONT_HERSHEY_COMPLEX, \
                        0.5, (0, 255, 255), 1, False)
        column_names = ['Frame', 'Person ID', 'Probability', 'Action Class']
        data = np.array(para)
        my_df = pd.DataFrame(data, columns=column_names)
        #you can change the output folder for saving the csv
        my_df.to_csv('/content/drive/My Drive/Activity_CSV.csv')
        #you can change the output folder for saving the frames
        cv2.imwrite('/content/drive/My Drive/Activity_Outputs/frames/%d.jpg' % count, frame)

def arg_parse():
    """
    Parse arguements to the detect module

    """


    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

      #change the video path accordingly
    parser.add_argument("--video", dest = 'video', help =
                        "Video to run detection upon",
                        default = "/content/Slow-Fast-pytorch-implementation/new.mp4", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)

    #change the path accordingly
    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "/content/Slow-Fast-pytorch-implementation/cfg/yolov3.cfg", type = str)

    #change the path accordingly
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "/content/Slow-Fast-pytorch-implementation/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float("0.4")
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80

    CUDA = torch.cuda.is_available()

    bbox_attrs = 5 + num_classes

    #change the path accordingly
    print("Loading network.....")
    model = Darknet("/content/Slow-Fast-pytorch-implementation/cfg/yolov3.cfg")
    model.load_weights("/content/Slow-Fast-pytorch-implementation/yolov3.weights")
    print("Network successfully loaded")

    #change the path accordingly
    print("load deep sort network....")
    deepsort = DeepSort("/content/Slow-Fast-pytorch-implementation/deep/checkpoint/ckpt.t7")
    print("Deep Sort Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()


    # model(get_test_input(inp_dim, CUDA), CUDA)

    model.eval()

    #######for sp detec##########

    #Initialize the model and change the path for the weights accordingly
    path_to_checkpoint = "/content/Slow-Fast-pytorch-implementation/slowfast_weight.pth"
    backbone_name = Config.BACKBONE_NAME
    backbone = BackboneBase.from_name(backbone_name)()
    model_sf = Model(backbone, 81, pooler_mode=Config.POOLER_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n=TrainConfig.RPN_PRE_NMS_TOP_N,
                  rpn_post_nms_top_n=TrainConfig.RPN_POST_NMS_TOP_N).cuda()
    model_sf.load(path_to_checkpoint)

    #Video file on which you want to run the model
    videofile = args.video
    cap = cv2.VideoCapture(videofile)

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    last = np.array([])
    last_time = time.time()

    start = time.time()

    #######for sp detec##########
    buffer = deque(maxlen=64)
    resize_width=400
    resize_height=300


    count=0
    ZZ = 0
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:

            #######for sp detec##########
            f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # will resize frames if not already final size
            f = cv2.resize(frame, (resize_width, resize_height))
            f=normalize(f)
            buffer.append(f)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(frame_height,frame_width)
            scale = [resize_width / frame_width, resize_height / frame_height]

            img, orig_im, dim = prep_image(frame, inp_dim)

            im_dim = torch.FloatTensor(dim).repeat(1,2)


            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1
                ZZ+= 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                print(ZZ)
                #cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue


            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)

            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2

            output[:,1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

            output = output.cpu().data.numpy()
            # print(output)
            bbox_xywh = output[:, 1:5]
            bbox_xywh[:,2] = bbox_xywh[:,2] - bbox_xywh[:,0]
            bbox_xywh[:,3] = bbox_xywh[:,3] - bbox_xywh[:,1]

            bbox_xywh[:, 0] = bbox_xywh[:, 0] + (bbox_xywh[:,2])/2
            bbox_xywh[:, 1] = bbox_xywh[:, 1] + (bbox_xywh[:, 3])/2
            cls_conf = output[:, 5]
            cls_ids = output[:, 7]

            if bbox_xywh is not None:
                mask = cls_ids == 0.0
                bbox_xywh = bbox_xywh[mask]
                cls_conf = cls_conf[mask]
                outputs = deepsort.update(bbox_xywh, cls_conf, orig_im)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]

            if len(buffer)==64:
                if count%3==0:
                    b=buffer
                    a = time.time()
                    b=np.array(b,dtype=np.float32)
                    print("time:", time.time() - a)
                    b = to_tensor(b)

                    image_batch=torch.tensor(b, dtype=torch.float).unsqueeze(0).cuda()
                    bbox_xyxy=np.array(bbox_xyxy,dtype=np.float)
                    bbox_xyxy[:, [0, 2]] *= scale[0]
                    bbox_xyxy[:, [1, 3]] *= scale[1]
                    detector_bboxes=torch.tensor(bbox_xyxy, dtype=torch.float).unsqueeze(0).cuda()

                    with torch.no_grad():
                        detection_bboxes, detection_classes, detection_probs = \
                            model_sf.eval().forward(image_batch, detector_bboxes_batch=detector_bboxes)

                    detection_bboxes=np.array(detection_bboxes.cpu())
                    detection_classes=np.array(detection_classes)
                    detection_probs=np.array(detection_probs)
                    #Get the corresponding classification label
                    detection_bboxes[:, [0, 2]] /= scale[0]
                    detection_bboxes[:, [1, 3]] /= scale[1]
                imshow(detection_bboxes,detection_classes,detection_probs,identities,count)
                count += 1


            #cv2.imshow("frame", orig_im)
            key = cv2.waitKey(0)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            ZZ += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            print(ZZ)

        else:
            break