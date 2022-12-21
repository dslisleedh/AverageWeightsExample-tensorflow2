import tensorflow as tf
from src.layers import *

import einops
from einops.layers.keras import Rearrange, Reduce

from tensorflow.python.util.tf_export import keras_export
import gin
from typing import Callable


@gin.configurable
class MLPMixer(tf.keras.models.Model):
    def __init__(
            self, intro_config: dict, feature_extractor_config: dict, classifier_config: dict
    ):
        super(MLPMixer, self).__init__()
        self.intro_config = intro_config
        self.feature_extractor_config = feature_extractor_config
        self.classifier_config = classifier_config

        self.intro = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=self.intro_config['filters'],
                kernel_size=self.intro_config['patch_size'],
                strides=self.intro_config['patch_size'],
                padding='valid',
                use_bias=False
            ),
            Rearrange('b h w c -> b (h w) c')
        ])
        n_blocks = self.feature_extractor_config.pop('n_blocks')
        drop_rates = self.feature_extractor_config.pop('drop_rates')
        drop_rates = tf.reshape(
            tf.linspace(0., drop_rates, n_blocks * 2), (n_blocks, 2)
        )
        self.feature_extractor = tf.keras.Sequential([
            MixerBlock(drop_rates[i], **self.feature_extractor_config) for i in range(n_blocks)
        ])
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.GlobalAvgPool1D(),
            tf.keras.layers.Dense(
                self.classifier_config['n_classes'], activation='softmax',
                kernel_initializer=tf.keras.initializers.Zeros()
            )
        ])

    def call(self, inputs, training: bool = False, *args, **kwargs):
        x = self.intro(inputs, training=training)
        x = self.feature_extractor(x, training=training)
        x = self.classifier(x, training=training)
        return x


@gin.configurable
class ViT(tf.keras.models.Model):
    def __init__(
            self, intro_config: dict, feature_extractor_config: dict, classifier_config: dict
    ):
        super(ViT, self).__init__()
        self.intro_config = intro_config
        self.feature_extractor_config = feature_extractor_config
        self.classifier_config = classifier_config

        self.intro = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=self.intro_config['filters'],
                kernel_size=self.intro_config['patch_size'],
                strides=self.intro_config['patch_size'],
                padding='valid',
                use_bias=False
            ),
            Rearrange('b h w c -> b (h w) c'),
            ClsToken()
        ])
        n_blocks = self.feature_extractor_config.pop('n_blocks')
        drop_rates = self.feature_extractor_config.pop('drop_rates')
        drop_rates = tf.reshape(
            tf.linspace(0., drop_rates, n_blocks * 2), (n_blocks, 2)
        )
        self.feature_extractor = tf.keras.Sequential([
            ViTBlock(drop_rates[i], **self.feature_extractor_config) for i in range(n_blocks)
        ])
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Lambda(lambda x: tf.gather(x, 0, axis=1)),
            tf.keras.layers.Dense(
                self.classifier_config['n_classes'], activation='softmax',
                kernel_initializer=tf.keras.initializers.Zeros()
            )
        ])

    def build(self, input_shape):
        self.intro.build(input_shape)
        self.positional_encoding = tf.Variable(
            tf.random.truncated_normal(stddev=.02, shape=(1,) + self.intro.output_shape[1:]),
            dtype=tf.float32, trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, training: bool = False, *args, **kwargs):
        x = self.intro(inputs, training=training)
        x += self.positional_encoding
        x = self.feature_extractor(x, training=training)
        x = self.classifier(x, training=training)
        return x
