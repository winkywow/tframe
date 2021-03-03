from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tframe.nets.net as tfr_net

from tframe import pedia
from tframe.core import single_input

from tframe.layers import Activation
from tframe.layers import Linear
from tframe.layers.layer import Layer
from tframe.layers.convolutional import Conv2D


class ResidualNet(tfr_net.Net):
  """Residual learning building block
      Ref: Kaiming He, etc. 'Deep Residual Learning for Image Recognition'.
      https://arxiv.org/abs/1512.03385"""
  def __init__(self, force_transform=False, **kwargs):
    # Call parent's constructor
    tfr_net.Net.__init__(self, 'res', **kwargs)

    # Initialize new fields
    self._force_transform = force_transform
    self._post_processes = []

    self._current_collection = self.children
    self._transform_layer = None

  # region : Properties

  def structure_string(self, detail=True, scale=True):
    body = tfr_net.Net.structure_string(self, detail, scale)
    result = 'sc({}){}'.format(
      body, '' if self._transform_layer is None else 't')

    # Add post process layers
    for layer in self._post_processes:
      assert isinstance(layer, Layer)
      result += '-> {}'.format(self._get_layer_string(layer, scale))

    # Return
    return result

  @property
  def structure_detail(self):
    # TODO: res_block structure_detail
    from tframe import hub
    from tframe.utils import stark
    import tframe.utils.format_string as fs

    widths = [33, 24, 20]
    rows = []
    add_to_rows = lambda cols: rows.append(fs.table_row(cols, widths))
    total_params, dense_total = 0, 0

    def get_num_string(num, dense_num):
      if num == 0: num_str = ''
      elif hub.prune_on or hub.etch_on:
        num_str = '{} ({:.1f}%)'.format(num, 100.0 * num / dense_num)
      else: num_str = str(num)
      return num_str

    for child in self.children + [self._transform_layer] + self._post_processes:
      if child is None:
        continue
      if isinstance(child, Layer):
        # Try to find variable in child
        # TODO: to be fixed
        # variables = [v for v in self.var_list if child.group_name in v.name]
        variables = [
          v for v in self.var_list
          if child.group_name == v.name.split('/')[self._level + 1]]
        num, dense_num = stark.get_params_num(variables, consider_prune=True)
        # Generate a row
        cols = [self._get_layer_string(child, True, True),
                child.output_shape_str, get_num_string(num, dense_num)]
        add_to_rows(cols)
      else:
        raise TypeError('!! unknown child type {}'.format(type(child)))

      # Accumulate total_params and dense_total_params
      total_params += num
      dense_total += dense_num
    return rows, total_params, dense_total

  # endregion : Properties

  # region : Abstract Implementation

  @single_input
  def _link(self, input_, **kwargs):
    """..."""
    assert isinstance(input_, tf.Tensor)

    # Link main part
    output = input_
    for layer in self.children:
      assert isinstance(layer, Layer)
      output = layer(output)

    # Shortcut
    input_shape = input_.get_shape().as_list()
    output_shape = output.get_shape().as_list()
    origin = input_
    if len(input_shape) != len(output_shape):
      raise ValueError('!! input and output must have the same dimension')
    if self._force_transform or input_shape != output_shape:
      if len(input_shape) == 2:
        # Add linear layer
        use_bias = self.kwargs.get('use_bias', False)
        self._transform_layer = Linear(
          output_dim=output_shape[1], use_bias=use_bias)
      elif len(input_shape) == 4:
        strides = 2 if input_shape[1] != output_shape[1] else 1
        self._transform_layer = Conv2D(filters=output_shape[-1], kernel_size=1,
                                       strides=strides)
      else: raise TypeError(
        '!! ResNet in tframe currently only support linear and Conv2D '
        'transformation.')
      # Save add
      self._transform_layer.full_name = self._get_new_name(
        self._transform_layer)
      origin = self._transform_layer(origin)
    # Do transformation
    output = output + origin

    # Link post process layers
    for layer in self._post_processes:
      assert isinstance(layer, Layer)
      if isinstance(layer, Activation): self._logits_tensor = output
      output = layer(output)

    # Return result
    return output

  # endregion : Abstract Implementation

  # region : Public Methods

  def add_shortcut(self):
    self._current_collection = self._post_processes

  def add(self, layer=None, **kwargs):
    if not isinstance(layer, Layer): raise TypeError(
        '!! layer added to ResNet block must be an instance of Layer')
    assert isinstance(self._current_collection, list)
    name = self._get_new_name(layer)
    layer.full_name = name
    self._current_collection.append(layer)

  # endregion : Public Methods
