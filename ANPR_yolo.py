import cv2
from matplotlib import pyplot as plt
import numpy as np
# import imutils
import easyocr
import tensorflow as tf

import Core.utils as utils
from Core.config import cfg
from Core.yolov4 import YOLOv4, decode

from absl import app, flags
from absl.flags import FLAGS

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


flags.DEFINE_string('input', 'inputs/1.jpg', 'path to input image')
def main(_argv):
    img = cv2.imread(FLAGS.input) # Reading input
    # img = cv2.imread('inputs/image7.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = 608
    
    STRIDES = np.array(cfg.YOLO.STRIDES)
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    XYSCALE = cfg.YOLO.XYSCALE
       
    def plateDetect(frame, input_size, model):
        '''Preprocesses image and pass it to
        trained model for license plate detection.
        Returns bounding box coordinates.
        '''
        frame_size = frame.shape[:2]
        image_data = utils.image_preprocess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
    
        pred_bbox = model.predict(image_data)
        pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
    
        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.25)
        bboxes = utils.nms(bboxes, 0.213, method='nms')
        
        return bboxes
    
    input_layer = tf.keras.layers.Input([size,size, 3])
    feature_maps = YOLOv4(input_layer, NUM_CLASS)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, NUM_CLASS, i)
        bbox_tensors.append(bbox_tensor)
    
    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model,'data/YOLOv4-obj_1000.weights')
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes = plateDetect(img,size, model) # License plate detection
     
    img = utils.draw_bbox(img, bboxes) # Draws bounding box around license plate
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    (x1, y1) = (bboxes[0][1], bboxes[0][0])
    (x2, y2) = (bboxes[0][3], bboxes[0][2])
    x1=np.floor(x1).astype('int')
    y1=np.floor(y1).astype('int')
    x2=np.floor(x2).astype('int')
    y2=np.floor(y2).astype('int')
    # print(img.shape)
    # print(x1,y1,x2,y2)
    cropped_image = gray[x1:x2, y1:y2]
    # cropped_image = gray[x1-3:x2+3, y1+5:y2-2]
    
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    # print(result)
    # fg1=fig.add_subplot(1,3,2)
    # plt.imshow(cropped_image)
    # plt.show()
    
    text0 = result[0][-2]
    print(text0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(img, text=text0, org=(50,50), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    # res = cv2.rectangle(img, , (0,255,0),2)
    # fg1=fig.add_subplot(1,3,3)
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.show()
    # cv2.imwrite('outs/wfrbd3.jpg', res) # Saving output

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass