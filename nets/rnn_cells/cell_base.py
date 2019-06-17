from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import hub
from tframe import pedia
from tframe.nets import RNet
from tframe.operators.apis.neurobase import RNeuroBase


class CellBase(RNet, RNeuroBase):
  """Base class for RNN cells.
     TODO: all rnn cells are encouraged to in inherit this class
  """
  net_name = 'cell_base'

  def __init__(
      self,
      activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      bias_initializer='zeros',
      layer_normalization=False,
      dropout_rate=0.0,
      zoneout_rate=0.0,
      **kwargs):

    # Call parent's constructor
    RNet.__init__(self, self.net_name)
    RNeuroBase.__init__(
      self,
      activation=activation,
      weight_initializer=weight_initializer,
      use_bias=use_bias,
      bias_initializer=bias_initializer,
      layer_normalization=layer_normalization,
      zoneout_rate=zoneout_rate,
      dropout_rate=dropout_rate,
      **kwargs)

    self._output_scale_ = None

  # region : Properties

  @property
  def _output_scale(self):
    if self._output_scale_ is not None: return self._output_scale_
    return self._state_size

  # TODO: this property is a compromise to avoid error in Net.
  @_output_scale.setter
  def _output_scale(self, val): self._output_scale_ = val

  @property
  def _scale_tail(self):
    assert self._state_size is not None
    return '({})'.format(self._state_size)

  def structure_string(self, detail=True, scale=True):
    return self.net_name + self._scale_tail if scale else ''

  # endregion : Properties

  def _get_s_bar(self, x, s, output_dim=None, use_reset_gate=False):
    if output_dim is None: output_dim = self._state_size
    if use_reset_gate:
      r = self.dense_rn(
        x, s, 'reset_gate', output_dim=self.get_dimension(s), is_gate=True)
      self._gate_dict['reset_gate'] = r
      s = r * s
    return self.dense_rn(x, s, 's_bar', self._activation, output_dim=output_dim)

  def _zoneout(self, new_s, prev_s, ratio):
    assert self.get_dimension(new_s) == self.get_dimension(prev_s)
    assert 0 < ratio < 1

    seed = tf.random_uniform(tf.shape(new_s), 0, 1)
    z = tf.cast(tf.less(seed, ratio), hub.dtype)
    zoned_out = z * prev_s + (1. - z) * new_s

    return tf.cond(tf.get_collection(
      pedia.is_training)[0], lambda: zoned_out, lambda: new_s)




