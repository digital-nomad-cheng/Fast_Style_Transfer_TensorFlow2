import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils import tf_utils

class UpSampleLayer(keras.layers.Layer):
    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]*2, input_shape[2]*2, input_shape[3]

    def get_config(self):
        base_config = super(UpSampleLayer, self).get_config()
        return base_config

    @classmethod
    def from_config(self):
        return cls(**config)

    def build(self, input_shape):
        super(UpSampleLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.image.resize(inputs, tf.shape(inputs)[1:3]*2, method=tf.image.ResizeMethod.BILINEAR)

class InstanceNormLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.epsilon = 1e-5
        super().__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape(shape[0], shape[1], shape[2], shape[3])

    def build(self, input_shape):
        depth = (input_shape[3],)
        self.scale = self.add_weight(shape=depth, name='gamma', initializer=keras.initializers.get('ones'))
        self.offset = self.add_weight(shape=depth, name='gamma', initializer=keras.initializers.get('ones'))
        super(InstanceNormLayer, self).build(input_shape)

    def call(self, inputs):
       mean, variance = tf.nn.moments(inputs, axes=[1,2], keepdims=True)
       inv = tf.math.rsqrt(variance + self.epsilon)
       normalized = (inputs - mean) * inv
       return self.scale * normalized + self.offset

class ReflectPadLayer(keras.layers.Layer):
    def __init__(self, pad_size, **kwargs):
        self.pad_size = pad_size
        super(ReflectPadLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([shape[0], shape[1] + self.pads_size*2, shape[2] + self.pad_size*2, shape[3]])

    def build(self, input_shape):
        super(ReflectPadLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.pad(inputs, [[0, 0], [self.pad_size, self.pad_size], [self.pad_size, self.pad_size], [0, 0]], "SYMMETRIC")

def inverted_res_block(inputs, stride, in_channels, out_channels, norm_layer=InstanceNormLayer, expansion=1):
    x = inputs
    x = keras.layers.Conv2D(expansion*in_channels, kernel_size=1, padding='same', use_bias=False, activation=None)(x)
    x = InstanceNormLayer()(x)
    x = tf.keras.layers.ReLU(6.)(x)
    x = ReflectPadLayer(1)(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='valid', use_bias=False, activation=None)(x)
    x = InstanceNormLayer()(x)
    x = keras.layers.ReLU(6.)(x)
    x = keras.layers.Conv2D(out_channels, kernel_size=1, padding='same', use_bias=False, activation=None)(x)
    x = InstanceNormLayer()(x)
    if in_channels == out_channels and stride == 1:
        return keras.layers.Add()([inputs, x])
    return x

def TransformerMobile():
    inputs = keras.layers.Input(shape=(256, 256, 3), dtype=tf.float32)
    x = ReflectPadLayer(4)(inputs)
    x = keras.layers.Conv2D(32, 9, 1, 'VALID', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = InstanceNormLayer()(x)
    x = keras.layers.Activation('relu')(x)    
    x = inverted_res_block(x, 2, 32, 64)
    x = inverted_res_block(x, 2, 64, 128)

    # residual block
    x = inverted_res_block(x, 1, 128, 128)
    x = inverted_res_block(x, 1, 128, 128)
    x = inverted_res_block(x, 1, 128, 128)
    x = inverted_res_block(x, 1, 128, 128)
    x = inverted_res_block(x, 1, 128, 128)
    
    # upsample layer
    x = keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)
    x = inverted_res_block(x, 1, 128, 64)
    x = keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)
    x = inverted_res_block(x, 1, 64, 32)
    
    x = ReflectPadLayer(4)(x)
    x = keras.layers.Conv2D(3, 9, 1, 'VALID', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    model = keras.models.Model(inputs, x)
    
    return model


if __name__ == "__main__":
    model = TransformerMobile()
    model.load_weights(tf.train.latest_checkpoint('./models'))
    print("successfully load model weights from './models")
    model.summary()
    img = Image.open('imgs/test/person.jpg')
    input_tensor = np.expand_dims(np.array(img), axis=0).astype('float32')
    output = model(input_tensor)
    print(output) 
    output = np.clip(np.squeeze(output, axis=0), 0, 255).astype('uint8')
    output_img = Image.fromarray(output)
    output_img.save('result.jpg')
    
