import gin
import tensorflow as tf
import tensorflow_datasets as tfds


@gin.configurable
def load_mnist_dataset():
    train_ds, valid_ds, test_ds = tfds.load(
        'mnist', as_supervised=True, split=['train[:80%]', 'train[80%:]', 'test'])
    return train_ds, valid_ds, test_ds


def preprocessing(x, y, training=False):
    x = tf.cast(x, tf.float32) / 255.0
    if training:
        b, h, w, c = x.shape
        x = tf.image.random_crop(x, (b, 25, 25, 1))
        x = tf.image.resize(x, (h, w))
    return x, y
