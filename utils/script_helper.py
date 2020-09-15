from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import re
from subprocess import call
from collections import OrderedDict

from tframe import checker
from tframe import console
from tframe.utils.local import re_find_single
from tframe.utils.misc import date_string
from tframe.utils import arg_parser
from tframe.configs.flag import Flag
from tframe.trainers import SmartTrainerHub

flags, flag_names = None, None


def register_flags(config_class):
  global flags, flag_names
  flags = [attr for attr in
           [getattr(config_class, key) for key in dir(config_class)]
           if isinstance(attr, Flag)]
  flag_names = [f.name for f in flags]


register_flags(SmartTrainerHub)


def check_flag_name(method):
  def wrapper(obj, flag_name, *args, **kwargs):
    assert isinstance(obj, Helper)
    if flag_name in obj.sys_keys: return
    # Make sure flag_name is not in parameter list of obj
    if flag_name in obj.param_keys:
      raise ValueError('!! Key `{}` has already been set'.format(flag_name))
    # Make sure flag_name is registered by tframe.Config
    if flag_name not in flag_names:
      print(
        ' ! `{}` may be an invalid flag, press [Enter] to continue ...'.format(
          flag_name))
      input()
    method(obj, flag_name, *args, **kwargs)

  return wrapper


class Helper(object):
  # Class variables
  true_and_false = (True, False)
  true = True
  false = False

  def __init__(self, module_name=None):
    self.module_name = module_name
    self._check_module()

    self.common_parameters = OrderedDict()
    self.hyper_parameters = OrderedDict()
    self.constraints = OrderedDict()
    self.bayes_optimizer_kwargs = OrderedDict()
    self.engine = None
    self.base_metric = None

    self._python_cmd = 'python' if os.name == 'nt' else 'python3'

    # System argv info
    self.sys_keys = []
    self._sys_runs = None
    self._add_script_suffix = None
    self._register_sys_argv()

  # region : Properties

  @property
  def command_head(self):
    return ['python', self.module_name] + [
      self._get_config_string(k, v) for k, v in self.common_parameters.items()]

  @property
  def param_keys(self):
    # Add keys from hyper-parameters
    keys = list(self.hyper_parameters.keys())
    # Add keys from common-parameters
    keys += list(self.common_parameters.keys())
    return keys

  @property
  def default_summ_name(self):
    script_name = re_find_single(r's\d+_\w+(?=.py)')
    return '{}_{}'.format(date_string(), script_name)

  # endregion : Properties

  # region : Public Methods

  def constrain(self, conditions, constraints):
    checker.check_type(conditions, dict)
    checker.check_type(constraints, dict)
    key = tuple([(k, v) for k, v in conditions.items()])
    self.constraints[key] = constraints

  @staticmethod
  def register_flags(config_class):
    register_flags(config_class)

  @check_flag_name
  def register(self, flag_name, *val):
    """Flag value can not be a tuple or a list"""
    if flag_name in self.sys_keys: return
    assert len(val) > 0
    if len(val) == 1 and isinstance(val[0], (tuple, list, str)):
      val = val[0]

    try:
      if isinstance(val, str): val_tmp = val
      else: val_tmp = val[0]
      p = arg_parser.Parser.parse(val_tmp, splitter='>')
      if p.name in ('bayes', 'bayers', 'Bayes'):
        self._register_bayes(flag_name, p, val)
        return
      else:
        raise Exception
    except:
      if isinstance(val, (list, tuple)) and len(val) > 1:
        self.hyper_parameters[flag_name] = val
      else:
        if isinstance(val, (list, tuple)): val = val[0]
        self.common_parameters[flag_name] = val
        self._show_flag_if_necessary(flag_name, val)

  def set_python_cmd_suffix(self, suffix='3'):
    self._python_cmd = 'python{}'.format(suffix)

  def run(self, times=1, engine='grid', save=False, mark='',
          bayes_kwargs=None):
    if self._sys_runs is not None:
      times = checker.check_positive_integer(self._sys_runs)
      console.show_status('Run # set to {}'.format(times))
    # Set the corresponding flags if save
    if save:
      self.common_parameters['save_model'] = True
    # Show parameters
    # TODO: show params_space
    self._show_parameters()

    # XXX
    if self.engine is None:
      self.engine = engine
    if self.engine in ('bayes', 'bxxxxxx'):
      self._run_bayes(save, mark, bayes_kwargs)
      return
    assert engine in ('grid', 'gridsearch')
    # Begin iteration
    counter = 0
    for run_id in range(times):
      history = []
      for hyper_dict in self._hyper_parameter_dicts():
        # Set counter here
        counter += 1
        # Grand self._add_script_suffix the highest priority
        if self._add_script_suffix is not None: save = self._add_script_suffix
        if save: self.common_parameters['script_suffix'] = '_{}{}'.format(
          mark, counter)

        params = self._get_all_configs(hyper_dict)
        self._apply_constraints(params)

        params_list = self._get_config_strings(params)
        params_string = ' '.join(params_list)
        if params_string in history: continue
        history.append(params_string)
        console.show_status(
          'Loading task ...', '[Run {}/{}]'.format(run_id + 1, times))
        call([self._python_cmd, self.module_name] + params_list)
        # call(self.command_head + params_list)
        print()

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def _show_flag_if_necessary(flag_name, value):
    if flag_name == 'gpu_id':
      console.show_status('GPU ID set to {}'.format(value))
    if flag_name == 'gather_summ_name':
      console.show_status('Notes will be gathered to `{}`'.format(value))

  def _apply_constraints(self, configs):
    assert isinstance(configs, dict)

    def _satisfy(condition):
      assert isinstance(condition, tuple) and len(condition) > 0
      for key, values in condition:
        if not isinstance(values, (tuple, list)): values = values,
        if key not in configs.keys():
          return False
        elif configs[key] not in values:
          return False
      return True

    def _set_flag(flag_name, value):
      if flag_name not in flag_names:
        raise KeyError(
          '!! Failed to set `{}`  since it has not been registered'.format(
            flag_name))
      configs[flag_name] = value

    for condition, constraint in self.constraints.items():
      assert isinstance(constraint, dict)
      if _satisfy(condition):
        for key, value in constraint.items(): _set_flag(key, value)

  @staticmethod
  def _get_config_string(flag_name, val):
    return '--{}={}'.format(flag_name, val)

  @staticmethod
  def _get_config_strings(config_dict):
    assert isinstance(config_dict, dict)
    return [Helper._get_config_string(key, val) for key, val in
            config_dict.items()]

  def _get_all_configs(self, hyper_dict):
    assert isinstance(hyper_dict, OrderedDict)
    all_configs = OrderedDict()
    all_configs.update(self.common_parameters)
    all_configs.update(hyper_dict)
    return all_configs

  def _check_module(self):
    """If module name is not provided, try to find one according to the
       recommended project organization"""
    if self.module_name is None:
      self.module_name = '../t{}.py'.format(
        re_find_single(r'(?<=s)\d+_\w+(?=.py)'))
      console.show_status('Module set to `{}`'.format(self.module_name))

    if not os.path.exists(self.module_name):
      raise AssertionError(
        '!! module {} does not exist'.format(self.module_name))

  def _hyper_parameter_dicts(self, keys=None):
    """Provide a generator of hyper-parameters for running"""
    if keys is None: keys = list(self.hyper_parameters.keys())
    if len(keys) == 0:
      yield OrderedDict()
    else:
      for val in self.hyper_parameters[keys[0]]:
        configs = OrderedDict()
        configs[keys[0]] = val
        for cfg_dict in self._hyper_parameter_dicts(keys[1:]):
          configs.update(cfg_dict)
          yield configs

  def _show_parameters(self):
    console.section('Parameters')

    def _show_config(name, od):
      assert isinstance(od, OrderedDict)
      if len(od) == 0: return
      console.show_info(name)
      for k, v in od.items(): console.supplement('{}: {}'.format(k, v), level=2)

    _show_config('Common Settings', self.common_parameters)
    _show_config('Hyper Parameters', self.hyper_parameters)
    _show_config('Constraints', self.constraints)
    print()

  def _register_sys_argv(self):
    for s in sys.argv[1:]:
      assert isinstance(s, str)
      # Check format
      r = re.fullmatch(r'--([\w_]+)=([-\w./,+:;]+)', s)
      if r is None: raise AssertionError(
        'Can not parse argument `{}`'.format(s))
      k, v = r.groups()
      assert isinstance(v, str)
      val_list = re.split(r'[,/]', v)
      # TODO: bayes parameters
      if k in ('run', 'runs'):
        assert len(val_list) == 1
        self._sys_runs = checker.check_positive_integer(int(val_list[0]))
        continue
      if k in ('save', 'brand'):
        assert len(val_list) == 1
        option = val_list[0]
        assert option.lower() in ('true', 'false')
        self._add_script_suffix = option.lower() == 'true'
        continue
      if k in ('random_state', 'random_seed'):
        assert len(val_list) == 1
        self.bayes_optimizer_kwargs['random_state'] = val_list[0]
        continue
      if k in ('engine', 'mode'):
        assert len(val_list) == 1
        self.engine = val_list[0]
        continue
      if k in ('search_iterations', 'search_iters', 'iters'):
        assert len(val_list) == 1
        self.bayes_optimizer_kwargs['search_iters'] = \
          checker.check_positive_integer(int(val_list[0]))
        continue
      if k in ('search_initial_points', 'initial_points'):
        assert len(val_list) == 1
        self.bayes_optimizer_kwargs['n_initial_points'] = \
          checker.check_positive_integer(int(val_list[0]))
        continue
      if k in ('metric', 'base_metric'):
        assert len(val_list) == 1
        self.base_metric = val_list[0]
        continue
      self.register(k, *val_list)
      self.sys_keys.append(k)

  # endregion : Private Methods

  # region : BayesSearch

  def _run_bayes(self, save=False, mark='', bayes_kwargs=None):
    from tframe.utils.param_search.param_space import get_random_seed
    if bayes_kwargs is None:
      bayes_kwargs = dict()
    assert isinstance(bayes_kwargs, dict)
    random_state = bayes_kwargs.get('random_state', 1234)
    search_initial_points = bayes_kwargs.get('search_initial_points', 2)
    search_iters = bayes_kwargs.get('search_iters', 10)
    search_iters = self.bayes_optimizer_kwargs.get('search_iters',
                                                   search_iters)
    if self.base_metric is None:
      self.base_metric = bayes_kwargs.get('metric', 'best_loss')

    optimizer_kwargs = {}
    history = []

    param_space = self.hyper_parameters
    optimizer_kwargs['random_state'] = random_state
    optimizer_kwargs['n_initial_points'] = search_initial_points

    optimizer_kwargs.update(self.bayes_optimizer_kwargs)
    optimizer_kwargs['random_state'] = get_random_seed(random_state)

    optimizer = self._make_optimizer(param_space, optimizer_kwargs)

    for param_id in range(search_iters):
      history = self._step_bayes(optimizer, param_space, param_id,
                                 search_iters, mark, save, history)

  def _make_optimizer(self, param_space, optimizer_kwargs):
    from tframe.utils.param_search.optimizer import Optimizer
    from tframe.utils.param_search.param_space import param_space_2list
    kwargs = optimizer_kwargs.copy()
    kwargs['params_space'] = param_space_2list(param_space)
    optimizer = Optimizer(**kwargs)
    return optimizer

  def _register_bayes(self, flag_name, parse, *val):
    from tframe.utils.param_search.param_space import (FloatParamSpace,
                                                       IntParamSpace,
                                                       CategoricalParamSpace)

    type = parse.get_kwarg('type', str)
    # TODO: categorical
    if type in ('categorical', 'c'):
      self.hyper_parameters[flag_name] = \
        CategoricalParamSpace(val[0][1:], transform='index')
    elif type in ('int', 'integer', 'i'):
      low = parse.get_kwarg('low', int)
      high = parse.get_kwarg('high', int)
      distribution = parse.get_kwarg('distribution', str, default='uniform')
      self.hyper_parameters[flag_name] = IntParamSpace(low, high, distribution,
                                                       transform='identity')
    elif type in ('float', 'f'):
      low = parse.get_kwarg('low', float)
      high = parse.get_kwarg('high', float)
      distribution = parse.get_kwarg('distribution', str, default='uniform')
      self.hyper_parameters[flag_name] = FloatParamSpace(low, high,
                                                         distribution,
                                                         transform='identity')
    else:
      print('!! bayes param\'s type should be int, float, categorical, got {}'
            .format(type))
      return

  def _step_bayes(self, optimizer, param_space, counter, iters, mark, save,
                  history):
    import pickle
    from tframe.utils.param_search.param_space import point_2dict
    from tframe import metrics as me
    params = optimizer.ask()
    # params = params[0]
    params_dict = point_2dict(param_space, params[0])

    # Grand self._add_script_suffix the highest priority
    if self._add_script_suffix is not None: save = self._add_script_suffix
    if save: self.common_parameters['script_suffix'] = '_{}{}'.format(
      mark, counter)

    try:
      params_config = self._get_all_configs(params_dict)
      self._apply_constraints(params_config)

      params_list = self._get_config_strings(params_config)
      params_string = ' '.join(params_list)
      # if params_string in history: return
      history.append(params_string)

      console.show_status(
        'Loading task ...', '[Run {}/{}]'.format(counter + 1, iters))
      call([self._python_cmd, self.module_name] + params_list)
      print()
    except:
      print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! get params error')
      print("params_config {}".format(params_config))
      print('params list {}'.format(params_list))
      print('params string {}'.format(params_string))
    # TODO: get results
    # smaller is better
    gather_summ_file = self.common_parameters.get('gather_summ_name',
                                                  (self.default_summ_name +
                                                   '.sum'))
    try:
      with open(gather_summ_file, 'rb') as f:
        self.notes = pickle.load(f)
      assert isinstance(self.notes, list)
    except:
      print('!! Failed to load {}'.format(gather_summ_file))
      return

    if self.base_metric not in self.notes[-1]._criteria:
      print('!! metric {} not found in Result Notes {}'.
            format(self.base_metric, self.notes[1].keys()))
    results = - self.notes[-1]._criteria[self.base_metric]
    # metric_quantity = me.get(self.base_metric)
    # if metric_quantity.lower_is_better:
    #   results = -results
    optimizer.tell(params, [results])
    return history

  # endregion : BayesSearch
