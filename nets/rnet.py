from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe import hub
from tframe.nets.net import Net


class RNet(Net):
  """Recurrent net which outputs states besides common result"""
  net_name = 'rnet'

  def __init__(self, name):
    # Call parent's constructor
    Net.__init__(self, name)

    # Attributes
    self._state_array = None
    self._state_size = None
    self._init_state = None

  # region : Properties

  @property
  def rnn_cells(self):
    assert self.is_root
    return [net for net in self.children if isinstance(net, RNet)]

  @property
  def rnn_cell_num(self):
    return len(self.rnn_cells)

  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    # Initiate init_state
    if self.is_root:
      states = []
      with tf.name_scope('InitStates'):
        for rnn_cell in self.rnn_cells:
          states.append(rnn_cell.init_state)
      assert len(states) == self.rnn_cell_num
      return tuple(states)
    else:
      # If initial state is a tuple, this property must be overriden
      assert self._state_size is not None
      # The initialization of init_state must be done under with_graph
      # .. decorator
      self._init_state = tf.placeholder(
        dtype=hub.dtype, shape=(None, self._state_size), name='init_state')
      return self._init_state

  # endregion : Properties

  # region : Overriden Methods

  def _link(self, pre_outputs, input_, **kwargs):
    """
    This methods should be called only by tf.scan
    :param pre_outputs: (output:Tensor, states:List of Tensors)
                          set None to use default initializer
    :param input_: data input in a time step
    :return: (output:Tensor, states:List of Tensors)
    """
    # Check inputs
    if pre_outputs is not None:
      assert isinstance(pre_outputs, tuple) and len(pre_outputs) == 2
      pre_states = pre_outputs[1]
      # The assertion below is not held by rnn_cells
      assert isinstance(pre_states, (tuple, list))
      assert len(pre_states) == self.rnn_cell_num
    else: pre_states = None
    assert isinstance(input_, tf.Tensor)

    # Link
    states = []
    output = input_
    state_cursor = 0
    for net in self.children:
      assert isinstance(net, Net)
      if isinstance(net, RNet):
        # rnn_cells in self.children accept state and input
        # .. and gives (output, state)
        output, state = net(pre_states[state_cursor], output)
        states.append(state)
        state_cursor += 1
      else:
        output = net(output)

    assert len(states) == len(pre_states)
    return output, tuple(states)

  # endregion : Overriden Methods

  # region : Public Methods

  def reset_state(self, batch_size):
    assert self.is_root
    self._state_array = self._get_zero_state(batch_size)

  # endregion : Public Methods

  # region : Private Methods

  def _get_zero_state(self, batch_size):
    if self.is_root:
      state_array = []
      for rnn_cell in self.rnn_cells:
        state_array.append(rnn_cell._get_zero_state(batch_size))
      return tuple(state_array)
    else:
      # If state is a tuple, this method must be overriden
      assert self._state_size is not None
      return np.zeros(shape=(batch_size, self._state_size))

  def _get_state_dict(self, batch_size=None):
    assert self.is_root
    state = (self._state_array if batch_size is None
             else self._get_zero_state(batch_size))
    return {self.init_state: state}

  # endregion : Private Methods

