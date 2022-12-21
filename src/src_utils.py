import gin

import tensorflow as tf
import tensorflow_addons as tfa


# Don't use this with EarlyStopping.
@gin.configurable
class SwapAverageWeights(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        optimizer_name = self.model.optimizer._name
        assert optimizer_name in ['MovingAverage', 'SWA'],\
            'Optimizer must be a MovingAverage optimizer, got {}'.format(optimizer_name)

    def on_test_begin(self, logs=None):
        self.shallow_weights = self.model.get_weights()
        self.model.optimizer.assign_average_vars(self.model.trainable_variables)

    def on_test_end(self, logs=None):
        self.model.set_weights(self.shallow_weights)

    def on_train_end(self, logs=None):
        self.model.optimizer.assign_average_vars(self.model.trainable_variables)


@gin.configurable
class SwapAverageWeightsEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, *args, **kwargs):
        super(SwapAverageWeightsEarlyStopping, self).__init__(*args, **kwargs)

    def on_train_begin(self, logs=None):
        optimizer_name = self.model.optimizer._name
        assert optimizer_name in ['MovingAverage', 'SWA'],\
            'Optimizer must be a MovingAverage optimizer, got {}'.format(optimizer_name)
        super(SwapAverageWeightsEarlyStopping, self).on_train_begin(logs)

    def on_test_begin(self, logs=None):
        self.shallow_weights = self.model.get_weights()
        self.model.optimizer.assign_average_vars(self.model.trainable_variables)

    def on_epoch_end(self, epoch, logs=None):
        super(SwapAverageWeightsEarlyStopping, self).on_epoch_end(epoch, logs)
        if not self.model.stop_training:
            self.model.set_weights(self.shallow_weights)


@gin.configurable
class SparseF1Score(tfa.metrics.F1Score):
    def __init__(self, *args, **kwargs):
        super(SparseF1Score, self).__init__(*args, **kwargs)
        self._num_classes = kwargs['num_classes']

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(y_true, self._num_classes)
        y_true = tf.gather(y_true, 0, axis=1)
        return super(SparseF1Score, self).update_state(y_true, y_pred, sample_weight)
