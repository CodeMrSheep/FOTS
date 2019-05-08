import tensorflow as tf
import recong
import roirotate
from dataReader import NUM_CLASSES
import cv2
import numpy as np
from tensorflow.contrib import slim

from nets import resnet_v1
tf.app.flags.DEFINE_float('keepProb', 0.8, '')
tf.app.flags.DEFINE_float('alpha', 1., '')
tf.app.flags.DEFINE_float('beta', 1., '')
FLAGS = tf.app.flags.FLAGS
features_stride = 4
text_scale = 224
class dect_model(object):
    def __init__(self,imgs,brboxes,reuse_variables=tf.AUTO_REUSE):
        """
        imgs: type:tensor, shape:[batchsize,224,224,3]

        sco_map:type:tensor, shape:[batchsize,56,56,1]
        geo_map:type:tensor, shape:[batchsize,56,56,4]
        theta_map:type:tensor, shape:[batchsize,56,56,1]
        featuremaps : type:tensor , shape:(batchsize,56,56,c)
        """
        self.reuse_variables = reuse_variables
        with tf.variable_scope(tf.get_variable_scope(), reuse=self.reuse_variables):
            self.featuremaps = self.model(imgs, is_training=True)[0]# resize to 224?
            self.sco_map = self.model(imgs,is_training=True)[1]
            self.geo_map = self.model(imgs,is_training=True)[2]
            self.theta_map = self.model(imgs,is_training=True)[3]
            #self.det = self.model(imgs,is_training=True)
            self.F_geometry = tf.concat([self.geo_map, self.theta_map], axis=-1)
            # self.rois, self.ws = roirotate.RoiRotate(self.featuremaps, features_stride, 32, 80,mode=0)(brboxes)


        if self.reuse_variables is None:
            org_rois, org_ws = roirotate.RoiRotate(imgs, features_stride, 32, 80,mode=0)(brboxes)
            # org_rois shape [b, 8, 64, 3]
            tf.summary.image('input', imgs)
            tf.summary.image('score_map_pred', self.sco_map * 255)
            tf.summary.image('geo_map_0_pred', self.F_geometry[:, :, :, 0:1])
            tf.summary.image('org_rois', org_rois, max_outputs=12)
    def unpool(self,inputs):
        return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])
    def mean_image_subtraction(self,images, means=[123.68, 116.78, 103.94]):
        num_channels = images.shape[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')
        channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=3, values=channels)
    def model(self,images, weight_decay=1e-5, is_training=True):
        images = self.mean_image_subtraction(images)

        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

        with tf.variable_scope('feature_fusion', values=[end_points.values]):
            batch_norm_params = {
                'decay': 0.997,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': is_training
            }
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(weight_decay)):
                f = [end_points['pool5'], end_points['pool4'],
                     end_points['pool3'], end_points['pool2']]
                for i in range(4):
                    print('Shape of f_{} {}'.format(i, f[i].shape))
                g = [None, None, None, None]
                h = [None, None, None, None]
                num_outputs = [None, 128, 64, 32]
                for i in range(4):
                    if i == 0:
                        h[i] = f[i]
                    else:
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], f[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= 2:
                        g[i] = self.unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
                feature_map = g[3]
                # here we use a slightly different way for regression part,
                # we first use a sigmoid to limit the regression range, and also
                # this is do with the angle map
                sco_map = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * text_scale
                angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                #F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return feature_map,sco_map,geo_map,angle_map
        #return Model(images,[sco_map, geo_map, theta_map,featuremaps])

    def dice_coefficient(self,y_true_sco, y_pred_sco,training_mask):
        '''
        dice loss
        :param y_true_cls:
        :param y_pred_cls:
        :param training_mask:
        :return:
        '''
        eps = 1e-5
        intersection = tf.reduce_sum(y_true_sco * y_pred_sco * training_mask)
        union = tf.reduce_sum(y_true_sco * training_mask) + tf.reduce_sum(y_pred_sco * training_mask) + eps
        loss = 1. - (2 * intersection / union)
        tf.summary.scalar('classification_dice_loss', loss)
        return loss

    def detect_and_recg_loss(self, det_trues, det_preds,det_mask,reg_trues,seq_preds,recg_mask):
        '''
        define the loss used for training, contraning two part,
        the first part we use dice loss instead of weighted logloss,
        the second part is the iou loss defined in the paper
        '''
        #det_trues:type:list shape(2,),content:(sco_map,Fuse_geometry(geo_map+theta_map))
        #det_trues:type:list shape(2,),content:(pred_sco_map,Fuse_geometry(geo_map+theta_map))
        classification_loss = self.dice_coefficient(det_trues[0], det_preds[0], det_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=det_trues[1], num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=det_preds[1], num_or_size_splits=5, axis=3)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -tf.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - tf.cos(theta_pred - theta_gt)
        tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * det_trues[0] * det_mask))
        tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * det_trues[0] * det_mask))
        L_g = L_AABB + 20 * L_theta
        self.detec_loss = tf.reduce_mean(L_g * det_trues[0] * det_mask) + classification_loss
        # self.recg_loss = recong_model.recong_loss(reg_trues,seq_preds,21,recg_mask)
        # self.DandRloss =  FLAGS.beta*self.detec_loss + FLAGS.alpha*recg_loss

        return self.detec_loss
    # def total_loss(self, score_maps, geo_maps, training_masks, btags, recg_masks):
    #     self.model_loss(score_maps, geo_maps, training_masks, btags, recg_masks)
    #     self.total_loss = tf.add_n([self.model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    #     if self.reuse_variables is None:
    #         tf.summary.image('score_map', score_maps)
    #         tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
    #         tf.summary.image('training_masks', training_masks)
    #     return self.total_loss, self.model_loss, self.detector_loss, self.recognizer_loss

