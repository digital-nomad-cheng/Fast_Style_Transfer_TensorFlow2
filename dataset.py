import glob

import tensorflow as tf

class Dataset:
    def __init__(self, dataset_path, img_size=None, batch_size=4):
        files = glob.glob(dataset_path+'/*')
        self.img_size = img_size
        self.num_imgs = len(files)
        print("Number of imgs in dataset:{0}".format(self.num_imgs))
        self.dataset = tf.data.Dataset.from_tensor_slices(files)
        self.dataset = self.dataset.shuffle(buffer_size=500)
        self.dataset = self.dataset.map(self.image_processing).batch(batch_size, drop_remainder=False)
        self.dataset = self.dataset.prefetch(buffer_size=500)

    def image_processing(self, filename, is_training=True):
        x = tf.io.read_file(filename)
        x = tf.image.decode_jpeg(x, channels=3)
        if self.img_size is not None:
            x = tf.image.resize(x, (self.img_size, self.img_size))
        return x

if __name__ == "__main__":
    dataset = Dataset('/home/ubuntu/dataset/COCO_2017/images', 256, 64)
    print(dataset.dataset)
