from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf

from tframe.models.model import Model
from tframe.nets.net import Net

from tframe import pedia
from tframe import FLAGS
from tframe import losses
from tframe import metrics
from tframe import TFData
from tframe import with_graph

from tframe.layers import Input
from tframe.layers import Linear


class VolterraNet(Model):
  """ A class for Volterra Networks"""

  def __init__(self, degree, depth, mark=None):
    # Check parameters
    if degree < 1: raise ValueError('!! Degree must be a positive integer')
    if depth < 0: raise ValueError('!! Depth must be a positive integer')

    # Call parent's constructor
    Model.__init__(self, mark)

    # Initialize fields
    self.degree = degree
    self.depth = depth
    self.T = {}
    self._input = Input([depth], name='input')
    self._output = None
    self._target = None

    # Initialize operators in each degree
    self._init_T()

  # region : Properties

  @property
  def linear_coefs(self):
    coefs = self._session.run(self.T[1].chain[0].chain[0].weights)
    return coefs.flatten()

  @property
  def operators(self):
    od = collections.OrderedDict()
    for i in range(1, self.degree + 1): od[i] = self.T[i]
    return od

  @property
  def description(self):
    result = ''
    for key, val in self.operators.items():
      assert isinstance(val, Net)
      result += 'T[{}]: {}\n'.format(key, val.structure_string())
    return result

  # endregion : Properties

  # region : Building

  @with_graph
  def build(self, loss='euclid', optimizer=None,
            metric=None, metric_name='Metric'):
    """Build model"""
    # Define output
    with tf.name_scope('Output'):
      self._output = tf.add_n([op() for op in self.T.values()], name='output')

    self._target = tf.placeholder(
      self._output.dtype, self._output.get_shape(), name='target')
    tf.add_to_collection(pedia.default_feed_dict, self._target)

    # Define loss
    loss_function = losses.get(loss)
    with tf.name_scope('Loss'):
      self._loss = loss_function(self._target, self._output)
      tf.summary.scalar('loss_sum', self._loss)
      # Try to add regularization loss
      reg_list = [op.regularization_loss for op in self.T.values()
                  if op.regularization_loss  is not None]
      reg_loss = None if len(reg_list) == 0 else tf.add_n(
        reg_list, name='reg_loss')
      self._loss = self._loss if reg_loss is None else self._loss + reg_loss

    # Define metric
    metric_function = metrics.get(metric)
    if metric_function is not None:
      pedia.memo[pedia.metric_name] = metric_name
      with tf.name_scope('Metric'):
        self._metric = metric_function(self._target, self._output)
        tf.summary.scalar('metric_sum', self._metric)

    # Define train step
    self._define_train_step(optimizer)

    # Print status and model structure
    self.show_building_info(
      **{'T[{}]'.format(key): val for key, val in self.operators.items()})

    # Launch session
    self.launch_model(FLAGS.overwrite and FLAGS.train)

    # Set built flag
    self._built = True

  # endregion : Building

  # region : Private Methods

  def _init_T(self):
    # Add empty nets to each degree
    for n in range(1, self.degree + 1):
      self.T[n] = Net('T{}'.format(n))
      self.T[n].add(self._input)

    # Initialize linear part
    self.T[1].add(Linear(output_dim=1))

  # endregion : Private Methods

  # region : Public Methods

  # TODO: Exactly the same as predict method in predictor.py
  def predict(self, data):
    # Sanity check
    if not isinstance(data, TFData):
      raise TypeError('!! Input data must be an instance of TFData')
    if not self.built: raise ValueError('!! Model not built yet')
    if self._session is None:
      self.launch_model(overwrite=False)

    if data.targets is None:
      outputs = self._session.run(
        self._output,
        feed_dict=self._get_default_feed_dict(data, is_training=False))
      return outputs
    else:
      outputs, loss = self._session.run(
        [self._output, self._loss],
        feed_dict=self._get_default_feed_dict(data, is_training=False))
      return outputs, loss

  # endregion : Public Methods

  """For some reason, do not delete this line"""

