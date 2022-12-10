import gin
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

from src.models import *
from src.layers import *
from src.src_utils import *
from utils.preprocessing import *
from utils.config import *

from typing import List, Union, Sequence
from functools import partial


def train(
        model: tf.keras.models, optimizer: Callable, loss: Callable,
        metrics: Sequence[Callable], patience: int, epochs: int, batch_size: int,
        datasets: Sequence[tf.data.Dataset]
):
    train_ds, valid_ds, test_ds = datasets
    preprocessing_augmentation = partial(preprocessing, training=True)
    train_ds = train_ds.shuffle(100000).batch(batch_size, drop_remainder=True).map(preprocessing_augmentation) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    valid_ds = valid_ds.batch(batch_size, drop_remainder=False).map(preprocessing) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=False).map(preprocessing) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = [
        tf.keras.callbacks.TensorBoard('./logs')
    ]
    es_kwargs = {
        'monitor': 'val_loss', 'patience': patience, 'restore_best_weights': True
    }
    if optimizer.get_config()['name'] in ['SWA', 'moving_average']:
        callbacks.append(
            SwapAverageWeightsEarlyStopping(**es_kwargs)
        )
    else:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(**es_kwargs)
        )
    model.fit(
        train_ds, validation_data=valid_ds, epochs=epochs,
        callbacks=callbacks
    )

    evaluation_result = model.evaluate(test_ds)
    result = dict(zip(model.metrics_names, evaluation_result))
    OmegaConf.save(OmegaConf.create(result), './result.yaml')
    return evaluation_result


@hydra.main(config_path='configs', config_name='config.yaml', version_base=None)
def main(main_config):
    @RunnerDecorator
    def _main():
        load_external_configs()
        config_files = [
            get_original_cwd() + f'/configs/models/{main_config["model"]}.gin',
            get_original_cwd() + f'/configs/hyper_params.gin',
        ]
        gin.parse_config_files_and_bindings(config_files, None)
        # Main config automatically saved by hydra
        with open(f'./sub_configs.gin', 'w') as f:
            f.write(gin.config_str())

        train_kwargs = load_train_components()
        train(**train_kwargs)

    _main()


if __name__ == '__main__':
    main()
