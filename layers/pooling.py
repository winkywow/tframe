from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

from tframe.utils import get_scale
from tframe.core.decorators import init_with_graph
from tframe.core.function import Function

from tensorflow.python.layers.pooling import MaxPool2D as MaxPool2D_
from tensorflow.python.layers.pooling import AvgPool2D as AvgPool2D_


class MaxPool2D(Layer, MaxPool2D_):
  """"""
  full_name = 'maxpool2d'
  abbreviation = 'maxpool'

  @init_with_graph
  def __init__(self, pool_size, strides,
               padding='same', data_format='channels_last',
               name=None, **kwargs):
    MaxPool2D_.__init__(
      self, pool_size, strides, padding, data_format, name, **kwargs)

  @property
  def structure_tail(self):
    size = lambda inputs: 'x'.join([str(n) for n in inputs])
    return '({}>{})'.format(size(self.pool_size), size(self.strides))

  @single_input
  def _link(self, input_=None, **kwargs):
    assert isinstance(input_, tf.Tensor)
    output = MaxPool2D_.__call__(self, input_, scope=self.full_name)
    # self.neuron_scale = get_scale(output)
    return output

  def __call__(self, *args, **kwargs):
    return Layer.__call__(self, *args, **kwargs)


class AvgPool2D(Layer, AvgPool2D_):
  full_name = 'avgpool2d'
  abbreviation = 'avgpool'
  
  @init_with_graph
  def __init__(self, pool_size, strides, padding='same', 
               data_format='channels_last', name=None, **kwargs):
    AvgPool2D_.__init__(self, pool_size=pool_size, strides=strides, 
                        padding=padding, data_format=data_format, name=name,
                        **kwargs)
  
  @property
  def structure_tail(self):
    size = lambda inputs: 'x'.join([str(n) for n in inputs])
    return '({}>{})'.format(size(self.pool_size), size(self.strides))
  
  @single_input
  def _link(self, input_=None, **kwargs):
    assert isinstance(input_, tf.Tensor)
    output = AvgPool2D_.__call__(self, input_, scope=self.full_name)
    return output
  
  def __call__(self, *args, **kwargs):
    return Layer.__call__(self, *args, **kwargs)
