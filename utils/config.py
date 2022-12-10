import tensorflow as tf
import tensorflow_addons as tfa

import gin
from gin.tf import external_configurables


def load_external_configs():
    gin.config.external_configurable(
        tf.nn.gelu, 'tf.nn.gelu'
    )

    gin.config.external_configurable(
        tfa.optimizers.MovingAverage, 'tfa.optimizers.MovingAverage'
    )
    gin.config.external_configurable(
        tfa.optimizers.SWA, 'tfa.optimizers.SWA'
    )
    gin.config.external_configurable(
        tfa.optimizers.AdamW, 'tfa.optimizers.AdamW'
    )
    gin.config.external_configurable(
        tf.keras.optimizers.Adam, 'tf.keras.optimizers.Adam'
    )

    def _register_losses(module):
        gin.config.external_configurable(module, module='tf.keras.losses')
    _register_losses(tf.keras.losses.MeanSquaredError)
    _register_losses(tf.keras.losses.MeanAbsoluteError)
    _register_losses(tf.keras.losses.BinaryCrossentropy)
    _register_losses(tf.keras.losses.SparseCategoricalCrossentropy)
    _register_losses(tf.keras.losses.CategoricalCrossentropy)

    def _register_metrics(module):
        gin.config.external_configurable(module, module='tf.keras.metrics')
    _register_metrics(tf.keras.metrics.MeanSquaredError)
    _register_metrics(tf.keras.metrics.MeanAbsoluteError)
    _register_metrics(tf.keras.metrics.Accuracy)
    _register_metrics(tf.keras.metrics.SparseCategoricalAccuracy)
    gin.config.external_configurable(tfa.metrics.F1Score, 'tfa.metrics.F1Score')


class RunnerDecorator:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        try:
            self.f()
            gin.clear_config()
        except Exception as e:
            print(e)
            gin.clear_config()
            raise e


@gin.configurable(name_or_fn='train_config')
def load_train_components(**kwargs) -> dict:
    return kwargs
