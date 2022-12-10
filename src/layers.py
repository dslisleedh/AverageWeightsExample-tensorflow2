import tensorflow as tf

import einops
from einops.layers.keras import Rearrange

from typing import Callable, Optional


class MLP(tf.keras.layers.Layer):
    def __init__(
            self, expansion_rate: int, use_bias: bool = False, activation: tf.nn = tf.nn.gelu
    ):
        super(MLP, self).__init__()
        self.expansion_rate = expansion_rate
        self.use_bias = use_bias
        self.activation = activation

    def build(self, input_shape):
        self.ln = tf.keras.layers.LayerNormalization()
        self.w1 = tf.keras.layers.Dense(
            input_shape[-1] * self.expansion_rate, use_bias=self.use_bias,
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.w2 = tf.keras.layers.Dense(
            input_shape[-1], use_bias=self.use_bias,
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        super(MLP, self).build(input_shape)

    def call(self, inputs, training: bool = False, *args, **kwargs):
        x = self.ln(inputs)
        x = self.w1(x)
        x = self.activation(x)
        x = self.w2(x)
        return inputs + x


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(
            self, n_heads: int, use_bias: bool = False
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.use_bias = use_bias

    def build(self, input_shape):
        self.ln = tf.keras.layers.LayerNormalization()
        self.to_qkv = tf.keras.layers.Dense(
            input_shape[-1] * 3, use_bias=self.use_bias,
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.to_out = tf.keras.layers.Dense(
            input_shape[-1], use_bias=self.use_bias,
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.scale = tf.Variable(
            tf.sqrt(tf.cast(input_shape[-1] / self.n_heads, tf.float32)),
            trainable=False
        )
        super(MultiHeadSelfAttention, self).build(input_shape)

    def call(self, inputs, training: bool = False, *args, **kwargs):
        x = self.ln(inputs)
        qkv = self.to_qkv(x)
        q, k, v = tf.unstack(
            einops.rearrange(qkv, 'b t (qkv h d) -> qkv b h t d', qkv=3, h=self.n_heads),
            3, axis=0
        )
        attn = tf.matmul(q, k, transpose_b=True) / self.scale
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.matmul(attn, v)
        out = einops.rearrange(out, 'b h t d -> b t (h d)')
        out = self.to_out(out)
        return inputs + out


class DropPath(tf.keras.layers.Layer):
    def __init__(
         self, forward: Callable, rate: float
    ):
        super(DropPath, self).__init__()
        self.forward = forward
        self.rate = rate

    def call(self, inputs, training: bool = False, *args, **kwargs):
        if not training or (self.rate == 0.):
            return self.forward(inputs, training=training)

        keep_prob = 1. - self.rate
        if tf.random.uniform((), minval=0., maxval=1.) < keep_prob:
            inputs = inputs + ((self.forward(inputs, training=training) - inputs) / keep_prob)

        return inputs


class MixerBlock(tf.keras.layers.Layer):
    def __init__(
            self, drop_rate: tf.Tensor,
            expansion_rate: int, use_bias: bool = False, activation: Optional[Callable] = tf.nn.gelu
    ):
        super(MixerBlock, self).__init__()
        self.drop_rate = drop_rate
        self.expansion_rate = expansion_rate
        self.use_bias = use_bias
        self.activation = activation

    def build(self, input_shape):
        self.spatial_MLP = DropPath(
            tf.keras.Sequential([
                Rearrange('b t c -> b c t'),
                MLP(self.expansion_rate, self.use_bias, self.activation),
                Rearrange('b c t -> b t c')
            ]), self.drop_rate[0]
        )
        self.channel_MLP = DropPath(
            MLP(self.expansion_rate, self.use_bias, self.activation), self.drop_rate[1]
        )
        super(MixerBlock, self).build(input_shape)

    def call(self, inputs, training: bool = False, *args, **kwargs):
        x = self.spatial_MLP(inputs, training, *args, **kwargs)
        x = self.channel_MLP(x, training, *args, **kwargs)
        return x


class ViTBlock(tf.keras.layers.Layer):
    def __init__(
            self, drop_rate: tf.Tensor,
            expansion_rate: int, n_heads: int, use_bias: bool = False, activation: Optional[Callable] = tf.nn.gelu
    ):
        super(ViTBlock, self).__init__()
        self.drop_rate = drop_rate
        self.expansion_rate = expansion_rate
        self.n_heads = n_heads
        self.use_bias = use_bias
        self.activation = activation

    def build(self, input_shape):
        self.MHSA = DropPath(
            MultiHeadSelfAttention(self.n_heads, self.use_bias), self.drop_rate[0]
        )
        self.MLP = DropPath(
            MLP(self.expansion_rate, self.use_bias, self.activation), self.drop_rate[1]
        )
        super(ViTBlock, self).build(input_shape)

    def call(self, inputs, training: bool = False, *args, **kwargs):
        x = self.MHSA(inputs, training, *args, **kwargs)
        x = self.MLP(x, training, *args, **kwargs)
        return x
