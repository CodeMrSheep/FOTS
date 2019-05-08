import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras.models import *
import numpy as np
import os

import roirotate
import recong
import dect_model


batchsize = 1
nb_class = 26+10+1
lr = 0.005
text_featuremaps_height = 32
text_featuremaps_max_width = 80
nb_featuremap_channel =32
max_steps = 10000
store_step = 50
maxlen = 21
# inp feed_dict
imgs = tf.placeholder(shape=(None,224,224,3),dtype=tf.float32)
sco_map_gt = tf.placeholder(shape=(None,56,56,1),dtype=tf.float32)
geo_map_gt = tf.placeholder(shape=(None,56,56,5),dtype=tf.float32)
det_trues = [sco_map_gt,geo_map_gt]
det_mask = tf.placeholder(shape=(None,56,56,1),dtype=tf.float32)
# recg_mask = tf.placeholder(shape=(None,),dtype=tf.float32,name="recg_mask")
brboxes=[]
for i in range(batchsize):
    outbox = tf.placeholder(shape=(None,4),dtype = tf.int32,name=str(i)+"outbox")
    cropbox = tf.placeholder(shape=(None,4),dtype = tf.int32,name=str(i)+"cropbox")
    angle = tf.placeholder(shape=(None,),dtype = tf.float32,name=str(i)+"angle")
    brboxes.append((outbox,cropbox,angle))
text_lable = tf.sparse_placeholder(tf.int32, name='text_lable')
# print(imgs)
# print(sco_map_gt)
# print(geo_map_gt)
# print(theta_map_gt)
# print(brboxes)
# print(text_lable)

#loss  function
# Model_detect_model = None
# features = None
M = dect_model.dect_model(imgs,brboxes)
features = M.featuremaps
sco_mag_pred = M.sco_map
geo_map_pred = M.F_geometry
det_preds = [sco_mag_pred,geo_map_pred]
#theta_map_pred = M.theta_map
# detect_loss = 0
text_featuremaps,ws = roirotate.RoiRotate(features=features,
                    features_stride=4,
                    fix_RoiHeight=text_featuremaps_height,
                    max_RoiWidth= text_featuremaps_max_width)(brboxes)

my_recong_model = recong.recong_model(text_featuremaps_height=text_featuremaps_height,
                                        text_featuremaps_max_width=text_featuremaps_max_width,
                                        nb_channel=nb_featuremap_channel,mode=0)
Model_recong_model = my_recong_model.model()
seq = Model_recong_model(text_featuremaps)
ws = tf.cast(tf.div(ws,4),tf.int32)
recong_loss = my_recong_model.recong_loss(sparse_lable= text_lable,seq = seq, sequence_length=ws)


dect_loss = M.detect_and_recg_loss(det_trues, det_preds,det_mask,text_lable,seq,1)
# total_loss, dect_loss, recong_loss
total_loss = tf.add(dect_loss,recong_loss,name="total_loss")


# solver & &sess init
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("*"*10)
print(tf.global_variables())
print("*"*10)
print(tf.gradients(total_loss,tf.global_variables()))
train_step=tf.train.AdamOptimizer(lr).minimize(total_loss)
#saver
saver = tf.train.Saver(tf.global_variables())

#generator,消费者，生产者模型
import dataReader
data_generator_vaild = dataReader.get_batch(num_workers=1,batch_size=batchsize,vis=False)

print("begin")
for step in range(1,max_steps):
    inp_dict = {}
    print(step)
    d_images, _, d_score_maps, d_geo_maps, d_training_masks, d_brboxes, d_btags, d_bRecgTags = next(
        data_generator_vaild)
    inp_dict[imgs] = np.array(d_images)
    inp_dict[sco_map_gt] = np.array(d_score_maps)
    inp_dict[geo_map_gt] = np.array(d_geo_maps)
    # inp_dict[theta_map_gt] = d_brboxes
    inp_dict[det_mask] = np.array(d_training_masks)
    # inp_dict[recg_mask] = np.array(d_bRecgTags)
    for j in range(batchsize):
        inp_dict[brboxes[j][0]] = d_brboxes[j][0]  # outBoxs
        inp_dict[brboxes[j][1]] = d_brboxes[j][1]  # cropBoxs
        inp_dict[brboxes[j][2]] = d_brboxes[j][2]  # angles

    cur_d_btags = [j for i in d_btags for j in i]
    cur_d_btags = my_recong_model.sparse_tuple_from_label(cur_d_btags)
    print(cur_d_btags)
    inp_dict[text_lable] = cur_d_btags

    # _, total_loss_value, detect_loss_value, recong_loss_value = \
    total_loss_value = sess.run([total_loss], inp_dict)
    print(step, "  ", total_loss_value)
    if step%store_step == 0:
        print(step,"  ", total_loss_value),#detect_loss_value,recong_loss_value)
        saver.save(sess, os.path.join('savemodel', 'model.ckpt'), global_step=step)