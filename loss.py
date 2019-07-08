import tensorflow as tf

@tf.function
def gram_matrix(x):
    shape_x = tf.shape(x)
    b, c = shape_x[0], shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)

@tf.function
def style_loss(style_gram, generated_gram):
    style_loss = tf.math.reduce_mean(tf.losses.mean_squared_error(style_gram, generated_gram))
    return style_loss

@tf.function 
def content_loss(content_imgs, generated_imgs):
    return tf.math.reduce_mean(tf.losses.mean_squared_error(content_imgs, generated_imgs))

@tf.function
def regularization_loss(x):
    loss = tf.math.reduce_sum(tf.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + tf.math.reduce_sum(tf.abs(x[:, :-1, :, :] - x[:, 1:, :, :]))
    return loss

if __name__ == "__main__":
    x = tf.random.uniform((1, 256, 256, 3))
    y = tf.random.uniform((1, 256, 256, 3))
    print(gram_matrix(x))
    print(style_loss(x, y))
    print(content_loss(x, y))
