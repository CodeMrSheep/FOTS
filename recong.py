import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras.models import *
import numpy as np

class recong_model():
    """
    对模型的实现，提供2种接口：__init__() 和 model()
    在__init__()中保存模型前向计算的结果seq，如self.seq
    在model()中，返回值是一个 Model类型的实例
    """

    def __init__(self, text_featuremaps_height=32,
                 text_featuremaps_max_width=100, nb_channel=512, nb_class=37,mode =0):
        """

        :param text_featuremaps_height: crnn的输入tensor的高度
        :param text_featuremaps_max_width: crnn的输入tensor的最大宽度
        :param nb_channel: crnn的输入tensor的通道数目
        :param nb_class: crnn的输出的unit个数，类别的概率分布
        :param mode: 上一个环节roirotate的模式，mode=0意味着 text_featuremap填充0到最大宽度
                                             mode=1意味着 text_featuremap保持原有的宽度
        """
        self.text_featuremaps_height = text_featuremaps_height
        self.text_featuremaps_max_weight = text_featuremaps_max_width
        self.nb_channel = nb_channel
        self.nb_class = nb_class

    def build(self, text_featuremaps, ws):
        """
        text_featuremaps: type: tensor, shape(batchsize,16,128,c) 输入的tensor的宽度需要被明确，否则在“bn,h,w,c = x.get_shape().as_list()；x = Reshape((w,c),input_shape=(h,w,c))(x)”会报错，所以设定一个最大值128.
        ws: type:tensor, shape(batchsize,)，记录了输入tensor的宽度的真实值，在“bn,h,w,c = x.get_shape().as_list()；x = Reshape((w,c),input_shape=(h,w,c))(x)”需要根据真实值做一个截断。

        seq : type:tensor ,shape:(batchsize,?,units)
        """
        ###继续用卷积提取特征
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='recong_model_block1_conv1')(text_featuremaps)
        x = MaxPooling2D(16, strides=(16, 1), name='recong_model_block1_pool')(x)

        ###Lstm
        bn, h, w, c = x.get_shape().as_list()
        x = Reshape((w, c), input_shape=(h, w, c))(x)
        print(x)
        x = Bidirectional(LSTM(256, return_sequences=True, name='recong_model_bilstm_2'))(x)
        x = Dense(123, activation='softmax', name='recong_model_output')(x)
        self.seq = x

    def sparse_tuple_from_label(self, sequences):
        """Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(0, len(seq), 1)))
            values.extend(seq)

        indices = np.asarray(indices)
        values = np.asarray(values)
        shape = np.array([len(sequences), np.asarray(indices).max(0)[1] + 1])
        return (indices, values, shape)


    def recong_loss(self, sparse_lable, seq,sequence_length,mask=1):
        """
        lable_trues: type: sparse tensor
        seq : type:tensor, shape:(batchsize,?,units)

        ctc_loss: type:tensor, shape:(1,)
        """
        ctc_loss = tf.nn.ctc_loss(labels = sparse_lable,inputs= seq,
                                  sequence_length=sequence_length,time_major=False,
                                  ignore_longer_outputs_than_inputs=True)
        cost = tf.reduce_mean(ctc_loss * mask)

        return cost

    def model(self):
        text_featuremaps = Input(
            shape=(self.text_featuremaps_height, self.text_featuremaps_max_weight, self.nb_channel))
        #         ws = Input(shape=(1))

        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(text_featuremaps)
        x = MaxPooling2D(2, strides=2, name='block1_pool')(x)  # 64x16x64

        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = MaxPooling2D(2, strides=2, name='block2_pool')(x)  # 128x8x32

        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = MaxPooling2D(2, strides=(2, 1), name='block3_pool')(x)
        x = ZeroPadding2D(padding=(0, 1))(x)  # 256x4x16

        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = MaxPooling2D(2, strides=(2, 1), name='block4_pool')(x)
        x = ZeroPadding2D(padding=(0, 1))(x)  # 512x2x42
        x = Conv2D(256, (2, 2), activation='relu', name='block5_conv1')(x)
        x = BatchNormalization()(x)  # 512x1x41

        #         bn,h,w,c = x.get_shape().as_list()
        #         x = Reshape((w,c),input_shape=(h,w,c))(x)
        x = Lambda(lambda x: tf.squeeze(x, axis=1))(x)

        # ==============================================RNN===============================================#
        x = Bidirectional(LSTM(256, return_sequences=True), name='bilstm_1')(x)
        x = Dense(256, activation='linear', name='fc_1')(x)
        x = Bidirectional(LSTM(256, return_sequences=True, name='bilstm_2'))(x)
        seq = Dense(self.nb_class, activation='softmax', name='output')(x)
        print(seq)
        return Model(text_featuremaps, seq)

if __name__ == "__main__":
    my_recong_model = recong_model(text_featuremaps_height=32,
                                          text_featuremaps_max_width=80,
                                          nb_channel=512)
    Model_recong_model = my_recong_model.model()

