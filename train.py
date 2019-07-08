import os
import argparse
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import vgg16
from transformer import TransformerMobile
from dataset import Dataset
import loss
from feature_extractor import create_feature_extractor


class Learner():
    def __init__(self, datase_path, style_img_path, img_size, style_layers, 
                content_layer_index, batch_size, learning_rate, s_weight, c_weight,
                r_weight, s_proportions):
        assert len(s_proportions) == len(style_layers)
        self.content_layer_index = content_layer_index
        self.content_dataset = Dataset('/home/ubuntu/dataset/COCO_2017/images', img_size[0], batch_size=batch_size)
        self.loss_net = create_feature_extractor(img_size, style_layers) 
        self.content_weight = c_weight
        self.style_weights = [s_weight*x for x in s_proportions]
        self.reg_weight = r_weight
        self.style_target = self.__prepare_style_target(style_img_path, batch_size)
    
    def train(self, epochs, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.5)
        transformer = TransformerMobile()
        for _ in range(epochs):
            print("epochs:", _)
            for i, x in tqdm(enumerate(self.content_dataset.dataset)):
                #if i == 10:
                #    break
                len_batch = len(x)
                # print("step:{0}".format(i))
                with tf.GradientTape() as tape:
                    y = transformer(x)
                    y = tf.keras.applications.vgg16.preprocess_input(y)
                    loss_layer_outputs = self.loss_net(y)
                    y_styles = tuple(loss.gram_matrix(loss_layer_output) for loss_layer_output in loss_layer_outputs)
                    y_content = loss_layer_outputs[self.content_layer_index]

                    x = tf.keras.applications.vgg16.preprocess_input(x)
                    loss_layer_outputs = self.loss_net(x)
                    x_content = loss_layer_outputs[self.content_layer_index]

                    content_loss = loss.content_loss(y_content, x_content)
                    content_loss *= self.content_weight

                    style_loss = 0.
                    for gm_y, gm_t, w in zip(y_styles, self.style_target, self.style_weights):

                        style_loss += w*loss.style_loss(gm_y, gm_t[:len_batch, :,:])

                    reg_loss = self.reg_weight*loss.regularization_loss(y)
                    
                    total_loss = style_loss + content_loss + reg_loss
                    print('style loss: {0}, content loss: {1}, reg loss: {2}, total loss:{3}'.format(style_loss, content_loss, reg_loss, total_loss))
                
                grads = tape.gradient(total_loss, transformer.trainable_variables)
                optimizer.apply_gradients(zip(grads, transformer.trainable_variables))
        transformer.save_weights(os.path.join('models', 'transformer'))

    
    def __prepare_style_target(self, img_path, batch_size):
        style_raw = tf.io.read_file(img_path)
        style_tensor = tf.image.decode_image(style_raw)
        style_tensor = tf.image.convert_image_dtype(style_tensor, tf.float32, saturate=False)*255.
        style_batch = tf.stack([style_tensor for _ in range(batch_size)], axis=0)
        style_batch = tf.keras.applications.vgg16.preprocess_input(style_batch)
        style_target = tuple(loss.gram_matrix(x) for x in  self.loss_net(style_batch))
        return style_target
        

if __name__ == "__main__":
    
    dataset_path = '/home/ubuntu/dataset/COCO_2017/images'
    style_img_path = '/home/ubuntu/dataset/COCO_2017/styles/style1_2.jpg'
    img_size = (128, 128)
    style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
    content_layer_index = 0
    batch_size=8
    learning_rate = 1e-3
    s_weight = 6*1e6
    c_weight = 1
    r_weight = 3*1e-7
    style_proportions = [.1, .1, .1, .7]
    lrn = Learner(dataset_path, style_img_path, img_size, style_layers, content_layer_index, 
                batch_size, learning_rate, s_weight, c_weight, r_weight, style_proportions)
    lrn.train(epochs=2, learning_rate=learning_rate)


