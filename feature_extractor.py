import tensorflow as tf
from tensorflow.keras.applications import vgg16

def create_feature_extractor(input_size, content_layers):
    
    if input_size is not None:
        h, w = input_size
        input_shape = (h, w, 3)
    else:
        input_shape = (256, 256, 3)
    outputs = []
    base_model = vgg16.VGG16(include_top=False, weights=None, input_shape=input_shape)
    for layer in base_model.layers:
        print(layer.name) 
    for layer in content_layers:
        outputs.append(base_model.get_layer(layer).output)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    model.load_weights('/home/ubuntu/ihandy/Fast-Style-Transfer-TensorFlow2.0/weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
    # freeze all the layers
    for layer in model.layers:
        layer.trainable = False
    # model.summary()
    return model
if __name__ == "__main__":
    create_feature_extractor((256, 256), ['block3_conv3', 'block4_conv3'])

        
