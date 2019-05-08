import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras.models import *
import numpy as np
import os


# tf.app.flags.DEFINE_integer('virtule_RoiHeight', 32, '')
# tf.app.flags.DEFINE_integer('virtule_MaxRoiWidth', 256, '')
# FLAGS = tf.app.flags.FLAGS

class RoiRotate(object):
    def __init__(self, features, features_stride, fix_RoiHeight, max_RoiWidth,mode=0):
        """

        :param features: detect模型的输出的feature map，一般是[batch,56,56,512]的tensor
        :param features_stride: feature map与原始输入图像[batch,224,224,3]的缩放比例，一般是4
        :param fix_RoiHeight: 经过ROIrotate后的text_featuremap的高度，一般为32
        :param max_RoiWidth: 经过ROIrotate后的text_featuremap的最大宽度，效果为rnn输入的序列长度，一般为
        :param mode: ROIrotate的模式：mode=0意味着 text_featuremap填充0到最大宽度
                                    mode=1意味着 text_featuremap保持原有的宽度
        """
        self.features = features
        self.features_stride = features_stride
        self.max_RoiWidth = np.int32(max_RoiWidth)
        self.fix_RoiHeight = np.int32(fix_RoiHeight)
        self.ratio = float(self.fix_RoiHeight) / self.max_RoiWidth
        self.mode = mode

    def scanFunc(self, ifeatures, outBox, cropBox, angle):
        #         print('*'*9,'scanFunc','*'*9)
        #         print(ifeatures)
        #         print(outBox)
        #         print(cropBox)
        #         print(angle)
        cropFeatures = tf.image.crop_to_bounding_box(ifeatures, outBox[1], outBox[0], outBox[3], outBox[2])
        rotateCropedFeatures = tf.contrib.image.rotate(cropFeatures, angle)
        textImgFeatures = tf.image.crop_to_bounding_box(rotateCropedFeatures, cropBox[1], cropBox[0], cropBox[3],
                                                        cropBox[2])
        #         print(cropFeatures)
        # resize keep ratio
        cnt1 = tf.divide(self.fix_RoiHeight, cropBox[3])
        cnt1 = tf.cast(cnt1, tf.float32)
        # 		w = tf.cast(tf.ceil(tf.multiply(cnt1,1.0*cropBox[2])),tf.int32)
        w = tf.cast(tf.ceil(tf.multiply(cnt1, tf.cast(cropBox[2], tf.float32))), tf.int32)
        resize_textImgFeatures = tf.image.resize_images(textImgFeatures, (self.fix_RoiHeight, w), 1)
        w = tf.minimum(w, self.max_RoiWidth)
        #         print("resize")
        #         print(resize_textImgFeatures)
        pad_or_crop_textImgFeatures = tf.image.crop_to_bounding_box(resize_textImgFeatures, 0, 0, self.fix_RoiHeight, w)
        if self.mode ==0:
            pad_or_crop_textImgFeatures = tf.image.pad_to_bounding_box(pad_or_crop_textImgFeatures, 0, 0, self.fix_RoiHeight, self.max_RoiWidth)
        return pad_or_crop_textImgFeatures

    def get_w(self, cropBox):
        cnt1 = tf.divide(self.fix_RoiHeight, cropBox[3])
        cnt1 = tf.cast(cnt1, tf.float32)

        w = tf.cast(tf.ceil(tf.multiply(cnt1, tf.cast(cropBox[2], tf.float32))), tf.int32)
        w = tf.minimum(w, self.max_RoiWidth)
        return w

    def __call__(self, brboxes, expand_w=20):
        paddings = tf.constant([[0, 0], [expand_w, expand_w], [expand_w, expand_w], [0, 0]])
        features_pad = tf.pad(self.features, paddings, "CONSTANT")
        features_pad = tf.expand_dims(features_pad, axis=1)
        # features_pad shape: [b, 1, h, w, c]
        nums = features_pad.shape[0]
        channels = features_pad.shape[-1]

        btextImgFeatures = []
        ws = []

        for b, rBoxes in enumerate(brboxes):
            outBoxes, cropBoxes, angles = rBoxes
            # outBoxes = tf.cast(tf.ceil(tf.divide(outBoxes, self.features_stride)), tf.int32)  # float div
            # cropBoxes = tf.cast(tf.ceil(tf.divide(cropBoxes, self.features_stride)), tf.int32) # float div

            outBoxes = tf.div(outBoxes, self.features_stride)  # int div
            cropBoxes = tf.div(cropBoxes, self.features_stride)  # int div

            outBoxes_xy = outBoxes[:, :2]
            outBoxes_xy = tf.add(outBoxes_xy, expand_w)
            outBoxes = tf.concat([outBoxes_xy, outBoxes[:, 2:]], axis=1)

            # len_crop = outBoxes.shape[0]  # error tf.stack cannot convert an unknown Dimension to a tensor: ?
            len_crop = tf.shape(outBoxes)[0]
            ifeatures_pad = features_pad[b]
            # ifeatures_tile = tf.tile(ifeatures_pad, tf.stack([len_crop, 1, 1, 1]))
            ifeatures_tile = tf.tile(ifeatures_pad, [len_crop, 1, 1, 1])
            #             print(ifeatures_tile)
            textImgFeatures = tf.map_fn(lambda x: self.scanFunc(x[0], x[1], x[2], x[3]),
                                        (ifeatures_tile, outBoxes, cropBoxes, angles), dtype=tf.float32)
            width_textImgFeatures = tf.map_fn(lambda x: self.get_w(x), cropBoxes, dtype=tf.int32)
            #             print('*'*5,'map_fn result','*'*5)
            #             print(textImgFeatures)
            #             print(width_textImgFeatures)
            ws.append(width_textImgFeatures)
            btextImgFeatures.append(textImgFeatures)

        ws = tf.concat(ws, axis=0)
        btextImgFeatures = tf.concat(btextImgFeatures, axis=0)
        return btextImgFeatures, ws