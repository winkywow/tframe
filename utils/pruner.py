from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import numpy as np
import tensorflow as tf

import tframe as tfr
from tframe.core import VariableSlot


class Pruner(object):

  def __init__(self, model):
    self._model = model
    self._dense_weights = []
    self._conv_filters = []

    # Plug in fractions
    self._dense_fraction = VariableSlot(self._model)
    with self._model.graph.as_default():
      self._dense_fraction.plug(tf.Variable(
        initial_value=-1.0, trainable=False, name='dense_fraction'))

    # Show status
    tfr.console.show_status('Pruner created <-<-<-<-<-<-<-<-<-=')

  # region : Properties

  @property
  def th(self): return tfr.hub

  @property
  def dense_fraction(self):
    return self._dense_fraction.fetch()

  @dense_fraction.setter
  def dense_fraction(self, value):
    self._dense_fraction.assign(value)

  # endregion : Properties

  # region : Public Methods

  def set_init_val(self):
    # Set dense fraction
    self.dense_fraction = 100.0
    # Set init_vals
    self._model.session.run([ws.assign_init for ws in self._dense_weights])

  def register_to_dense(self, weights):
    slot = WeightSlot(weights)
    self._dense_weights.append(slot)
    return slot.masked_weights

  def clear(self):
    raise NotImplementedError

  @staticmethod
  def extractor(*args):
    if not tfr.hub.prune_on or not tfr.hub.export_masked_weights: return
    pruner = tfr.context.pruner
    for i, slot in enumerate(pruner._dense_weights):
      key = 'weights_{}'.format(i + 1)
      tfr.context.variables_to_export[key] = slot.masked_weights

  # endregion : Public Methods

  # region : Prune and save

  def prune_and_save(self):
    self.prune()
    self.save_next_model()

  def prune(self):
    # Get percentage
    assert self.th.prune_on
    p = tfr.hub.pruning_rate_fc
    assert 0 < p < 1.0
    # Update fraction
    self.dense_fraction = self.dense_fraction * (1 - p)
    # Fetch, prune and set
    self._fetch_masked_weights()
    tfr.console.show_status('Masked weights fetched. Pruning ...')
    self._set_weights_and_masks(self.dense_fraction)
    # Show status
    tfr.console.show_status(
      'Dense fraction decreased to {:.2f}'.format(self.dense_fraction))

  def save_next_model(self):
    from tframe.models.model import Model
    model = self._model
    assert isinstance(model, Model)
    # Reset metrics
    model.metric.record = -1
    model.metric.mean_record = -1
    # Reset counter
    model.counter = 0
    # Update mark
    iteration = self.th.pruning_iterations
    if iteration < 10: k = -1
    else:
      assert iteration >= 10
      k = -2
    assert int(model.mark[k:]) == iteration
    model.mark = model.mark[:k] + str(iteration + 1)
    model.agent.save_model()
    # Show status
    tfr.console.show_status('Model {} saved.'.format(model.mark))

  # endregion : Prune and save

  # region : Private Methods

  def _run_op(self, ops):
    from tframe.models.model import Model
    model = self._model
    assert isinstance(model, Model)
    return model.session.run(ops)

  def _fetch_masked_weights(self):
    masked_weights = [ws.masked_weights for ws in self._dense_weights]
    value_masked_weights = self._run_op(masked_weights)
    for vmw, ws in zip(value_masked_weights, self._dense_weights):
      assert isinstance(ws, WeightSlot)
      ws.value_masked_weights = vmw

  def _set_weights_and_masks(self, fraction):
    reset_weights_ops = [ws.reset_weights for ws in self._dense_weights]
    set_mask_ops = [
      ws.get_assign_mask_op(fraction) for ws in self._dense_weights]
    self._run_op([reset_weights_ops, set_mask_ops])
    # Show status
    tfr.console.show_status('Weights reset and masks updated.')

  # endregion : Private Methods


class WeightSlot(object):
  """Weight Slot is not a tframe Slot"""

  def __init__(self, weights):
    assert isinstance(weights, tf.Variable)
    self.weights = weights
    self.init_val = tf.Variable(
      tf.zeros_like(weights), trainable=False, name='init_val')
    self.mask = tf.Variable(
      tf.ones_like(weights), trainable=False, name='mask')
    self.masked_weights = tf.multiply(self.weights, self.mask)

    # Define assign ops
    self.assign_init = tf.assign(self.init_val, self.weights)
    self.reset_weights = tf.assign(self.weights, self.init_val)

    # Define weights and mask placeholder
    self.value_masked_weights = None

  def get_assign_mask_op(self, weight_fraction):
    w = self.value_masked_weights
    assert isinstance(w, np.ndarray)
    assert 0 < weight_fraction < 100
    weight_fraction = np.ceil(weight_fraction)
    # Get weights magnitude
    w = np.abs(w)
    # Create mask
    mask = np.zeros_like(w, dtype=np.float32)
    mask[w > np.percentile(w, 100 - weight_fraction)] = 1.0
    # Return mask assign operator
    op = tf.assign(self.mask, mask)
    return op




