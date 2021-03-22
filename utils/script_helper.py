from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import re
import time
from subprocess import run
from collections import OrderedDict

from tframe import console
from tframe.utils.note import Note
from tframe.utils.local import re_find_single
from tframe.utils.misc import date_string
# <<<<<<< HEAD
# from tframe.utils import arg_parser
# =======
from tframe.utils.file_tools.imp import import_from_path
from tframe.utils.string_tools import get_time_string
from tframe.utils.file_tools.io_utils import safe_open
from tframe.utils.organizer.task_tools import update_job_dir
# >>>>>>> 2c2bb62db734310d5ab5fa0cb66e970e161ddebc
from tframe.configs.flag import Flag
from tframe.trainers import SmartTrainerHub

from tframe.alchemy.pot import Pot
from tframe.alchemy.scrolls import get_argument_keys

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
    if flag_name in obj.sys_flags: return
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

  BAYESIAN = 'BAYESIAN'
  GRID_SEARCH = 'GRID_SEARCH'

  class CONFIG_KEYS(object):
    add_script_suffix = 'add_script_suffix'
    auto_set_hp_properties = 'auto_set_hp_properties'
    strategy = 'strategy'
    criterion = 'criterion'
    greater_is_better = 'greater_is_better'

  def __init__(self, module_name=None):
    self.module_name = module_name
    self._check_module()

    self.pot = Pot(self._get_summary)

    self.common_parameters = OrderedDict()
    self.hyper_parameters = OrderedDict()
    self.constraints = OrderedDict()
    self.bayes_optimizer_kwargs = OrderedDict()
    self.engine = None
    self.base_metric = None

    self._python_cmd = 'python' if os.name == 'nt' else 'python3'
    self._root_path = None

    # System argv info. 'sys_keys' will be filled by _register_sys_argv.
    # Any config registered by Helper.register method with 1st arg in
    #  this list will be ignored. That is, system args have the highest priority
    # USE SYS_CONFIGS WITH DATA TYPE CONVERSION!
    self.sys_flags = []
    self.config_dict = OrderedDict()
    self._init_config_dict()
    self._register_sys_argv()

  # region : Properties

  @property
  def configs(self):
    od = OrderedDict()
    for k, v in self.config_dict.items():
      if v is not None: od[k] = v
    return od

  @property
  def hyper_parameter_keys(self):
    return self.pot.hyper_parameter_keys

  @property
  def command_head(self):
    return ['python', self.module_name] + [
      self._get_hp_string(k, v) for k, v in self.common_parameters.items()]

  @property
  def param_keys(self):
    # Add keys from hyper-parameters
    keys = self.hyper_parameter_keys
    # Add keys from common-parameters
    keys += list(self.common_parameters.keys())
    return keys

  @property
  def default_summ_name(self):
    script_name = re_find_single(r's\d+_\w+(?=.py)')
    return '{}_{}'.format(date_string(), script_name)

  @property
  def summ_file_name(self):
    key = 'gather_summ_name'
    assert key in self.common_parameters
    return self.common_parameters[key]

  @property
  def shadow_th(self):
    task = import_from_path(self.module_name)
    return task.core.th

  @property
  def root_path(self):
    if self._root_path is not None: return self._root_path
    task = import_from_path(self.module_name)
    update_job_dir(task.id, task.model_name)
    self._root_path = task.core.th.job_dir
    return self.root_path

  # endregion : Properties

  # region : Public Methods

  def constrain(self, conditions, constraints):
    # Make sure hyper-parameter keys have been registered.
    for key in list(conditions.keys()) + list(constraints.keys()):
      if key not in flag_names: raise KeyError(
          '!! Failed to set `{}`  since it has not been registered'.format(key))
    self.pot.constrain(conditions, constraints)

  @staticmethod
  def register_flags(config_class):
    register_flags(config_class)

  @check_flag_name
  def register(self, flag_name, *val, hp_type=None, scale=None):
    """Flag value can not be a tuple or a list"""
    assert len(val) > 0
# <<<<<<< HEAD
#     if len(val) == 1 and isinstance(val[0], (tuple, list, str)):
#       val = val[0]
#
#     try:
#       if isinstance(val, str): val_tmp = val
#       else: val_tmp = val[0]
#       p = arg_parser.Parser.parse(val_tmp, splitter='>')
#       if p.name in ('bayes', 'bayers', 'Bayes'):
#         self._register_bayes(flag_name, p, val)
#         return
#       else:
#         raise Exception
#     except:
#       if isinstance(val, (list, tuple)) and len(val) > 1:
#         self.hyper_parameters[flag_name] = val
#       else:
#         if isinstance(val, (list, tuple)): val = val[0]
#         self.common_parameters[flag_name] = val
#         self._show_flag_if_necessary(flag_name, val)
# =======
    if len(val) == 1 and isinstance(val[0], (tuple, list)): val = val[0]

    if isinstance(val, (list, tuple)) and len(val) > 1:
      if flag_name in self.sys_flags:
        self.pot.set_hp_properties(flag_name, hp_type, scale)
      else: self.pot.register_category(flag_name, val, hp_type, scale)
    else:
      if flag_name in self.sys_flags: return
      if isinstance(val, (list, tuple)): val = val[0]
      self.common_parameters[flag_name] = val
      self._show_flag_if_necessary(flag_name, val)
# >>>>>>> 2c2bb62db734310d5ab5fa0cb66e970e161ddebc

  def set_hp_property(self, name, hp_type, scale=None):
    assert name in self.pot.hyper_parameter_keys
    self.pot.set_hp_properties(name, hp_type, scale)

  def configure(self, **kwargs):
    for k, v in kwargs.items():
      assert not isinstance(v, (tuple, list, set))
      if k not in self.config_dict:
        raise KeyError('!! Unknown system config `{}`'.format(k))
      # Set only when the corresponding config has not been set by
      #   system arguments
      if self.config_dict[k] is None: self.config_dict[k] = v

  def set_python_cmd_suffix(self, suffix='3'):
    self._python_cmd = 'python{}'.format(suffix)

# <<<<<<< HEAD
#   def run(self, times=1, engine='grid', save=False, mark='',
#           bayes_kwargs=None, rehearsal=False):
#     if self._sys_runs is not None:
#       times = checker.check_positive_integer(self._sys_runs)
#       console.show_status('Run # set to {}'.format(times))
#     # Set the corresponding flags if save
#     if save:
#       self.common_parameters['save_model'] = True
#     # Show parameters
#     # TODO: show params_space
#     self._show_parameters()
#
#     # XXX
#     if self.engine is None:
#       self.engine = engine
#     if self.engine in ('bayes', 'bxxxxxx'):
#       self._run_bayes(save, mark, bayes_kwargs)
#       return
#     assert engine in ('grid', 'gridsearch')
#     # Begin iteration
#     counter = 0
#     for run_id in range(times):
#       history = []
#       for hyper_dict in self._hyper_parameter_dicts():
#         # Set counter here
#         counter += 1
#         # Grand self._add_script_suffix the highest priority
#         if self._add_script_suffix is not None: save = self._add_script_suffix
#         if save: self.common_parameters['script_suffix'] = '_{}{}'.format(
#           mark, counter)
#
#         params = self._get_all_configs(hyper_dict)
#         self._apply_constraints(params)
#
#         params_list = self._get_config_strings(params)
#         params_string = ' '.join(params_list)
#         if params_string in history: continue
#         history.append(params_string)
#         console.show_status(
#           'Loading task ...', '[Run {}/{}][{}]'.format(
#             run_id + 1, times, len(history)))
#         console.show_info('Hyper-parameters:')
#         for k, v in hyper_dict.items():
#           console.supplement('{}: {}'.format(k, v))
#         if not rehearsal:
#           call([self._python_cmd, self.module_name] + params_list)
#           print()
#       # End of the run
#       if rehearsal: return
# =======
  def run(self, strategy='grid', rehearsal=False, **kwargs):
    """Run script using the given 'strategy'. This method is compatible with
       old version of tframe script_helper, and should be deprecated in the
       future. """
    # Show section
    console.section('Script Information')
    # Show pot configs
    self._show_dict('Pot Configurations', self.configs)
    # Hyper-parameter info will be showed when scroll is set
    self.configure(**kwargs)
    # Do some auto configuring, e.g., set greater_is_better based on the given
    #   criterion
    self._auto_config()
    self.pot.set_scroll(self.configs.get('strategy', strategy), **self.configs)
    # Show common parameters
    self._show_dict('Common Settings', self.common_parameters)
# >>>>>>> 2c2bb62db734310d5ab5fa0cb66e970e161ddebc

    # Begin iteration
    for i, hyper_params in enumerate(self.pot.scroll.combinations()):
      # Show hyper-parameters
      console.show_info('Hyper-parameters:')
      for k, v in hyper_params.items():
        console.supplement('{}: {}'.format(k, v), level=2)
      # Run process if not rehearsal
      if rehearsal: continue
      console.split()
      # Export log if necessary
      if self.pot.logging_is_needed: self._export_log()
      # Run
      self._run_process(hyper_params, i)

  # endregion : Public Methods

  # region : Private Methods

  def _init_config_dict(self):
    """Config keys include that specified in self.CONFIG_KEYS and
       the primary arguments in the constructor of sub-classes of Scroll"""
    assert isinstance(self.config_dict, OrderedDict)
    # Add all keys specified in CONFIG_KEYS
    key_list = [k for k in self.CONFIG_KEYS.__dict__ if k[:2] != '__']
    # Get add keys from scroll classes
    key_list += get_argument_keys()
    # Initialize these configs as None
    # key_list may contain duplicated keys, which is OK
    for key in key_list: self.config_dict[key] = None

  def _auto_config(self):
    if self.configs.get(self.CONFIG_KEYS.strategy, 'grid') == 'grid': return
    # Try to automatically set greater_is_better
    if self.CONFIG_KEYS.greater_is_better not in self.configs:
      criterion = self.config_dict[self.CONFIG_KEYS.criterion]
      if isinstance(criterion, str):
        criterion = criterion.lower()
        if any(['accuracy' in criterion, 'f1' in criterion]):
          self.config_dict[self.CONFIG_KEYS.greater_is_better] = True
        elif any(['loss' in criterion, 'cross_entropy' in criterion]):
          self.config_dict[self.CONFIG_KEYS.greater_is_better] = False

    # Try to set hyper-parameters' properties
    if self.configs.get(self.CONFIG_KEYS.auto_set_hp_properties, True):
      self.auto_set_hp_properties()

  def _run_process(self, hyper_params, index):
    assert isinstance(hyper_params, dict)
    # Handle script suffix option
    if self.configs.get('add_script_suffix', False):
      self.common_parameters['script_suffix'] = '_{}'.format(index + 1)
    # Run
    configs = self._get_all_configs(hyper_params)
    cmd = [self._python_cmd, self.module_name] + self._get_hp_strings(configs)
    run(cmd)
    print()

  @staticmethod
  def _show_flag_if_necessary(flag_name, value):
    if flag_name == 'gpu_id':
      console.show_status('GPU ID set to {}'.format(value))
    if flag_name == 'gather_summ_name':
      console.show_status('Notes will be gathered to `{}`'.format(value))

  @staticmethod
  def _get_hp_string(flag_name, val):
    return '--{}={}'.format(flag_name, val)

  @staticmethod
  def _get_hp_strings(config_dict):
    assert isinstance(config_dict, dict)
    return [Helper._get_hp_string(key, val) for key, val in
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

# <<<<<<< HEAD
#   def _hyper_parameter_dicts(self, keys=None):
#     """Provide a generator of hyper-parameters for running"""
#     if keys is None: keys = list(self.hyper_parameters.keys())
#     if len(keys) == 0:
#       yield OrderedDict()
#     else:
#       for val in self.hyper_parameters[keys[0]]:
#         configs = OrderedDict()
#         configs[keys[0]] = val
#         for cfg_dict in self._hyper_parameter_dicts(keys[1:]):
#           configs.update(cfg_dict)
#           yield configs
#
#   def _show_parameters(self):
#     console.section('Parameters')
#
#     def _show_config(name, od):
#       assert isinstance(od, OrderedDict)
#       if len(od) == 0: return
#       console.show_info(name)
#       for k, v in od.items(): console.supplement('{}: {}'.format(k, v), level=2)
#
#     _show_config('Common Settings', self.common_parameters)
#     _show_config('Hyper Parameters', self.hyper_parameters)
#     _show_config('Constraints', self.constraints)
#     print()
# =======
  @staticmethod
  def _show_dict(name, od):
    assert isinstance(od, OrderedDict)
    if len(od) == 0: return
    console.show_info(name)
    for k, v in od.items():
      console.supplement('{}: {}'.format(k, v), level=2)
# >>>>>>> 2c2bb62db734310d5ab5fa0cb66e970e161ddebc

  def _register_sys_argv(self):
    """When script 'sX_YYYY.py' is launched using command line tools, system
       arguments other than tframe flags are allowed. These arguments, like
       tframe flags arguments passed via command line, also have the highest
       priority that will overwrite the corresponding arguments (if any) defined
       in related python modules.

       TODO: this method should be refactored
    """
    # Check each system arguments
    for s in sys.argv[1:]:
      assert isinstance(s, str)
      # Check format (pattern: --flag_name=value)
      r = re.fullmatch(r'--([\w_]+)=([-\w./,+:;]+)', s)
      if r is None: raise AssertionError(
        'Can not parse argument `{}`'.format(s))
      # Parse key and value
      k, v = r.groups()
      assert isinstance(v, str)
      val_list = re.split(r'[,/]', v)
# <<<<<<< HEAD
#       # TODO: bayes parameters
#       if k in ('run', 'runs'):
#         assert len(val_list) == 1
#         self._sys_runs = checker.check_positive_integer(int(val_list[0]))
#         continue
#       if k in ('save', 'brand'):
# =======
      # Check system configurations
      if k in self.config_dict:
# >>>>>>> 2c2bb62db734310d5ab5fa0cb66e970e161ddebc
        assert len(val_list) == 1
        self.config_dict[k] = val_list[0]
        continue
# <<<<<<< HEAD
#       if k in ('random_state', 'random_seed'):
#         assert len(val_list) == 1
#         self.bayes_optimizer_kwargs['random_state'] = val_list[0]
#         continue
#       if k in ('engine', 'mode'):
#         assert len(val_list) == 1
#         self.engine = val_list[0]
#         continue
#       if k in ('search_iterations', 'search_iters', 'iters'):
#         assert len(val_list) == 1
#         self.bayes_optimizer_kwargs['search_iters'] = \
#           checker.check_positive_integer(int(val_list[0]))
#         continue
#       if k in ('search_initial_points', 'initial_points'):
#         assert len(val_list) == 1
#         self.bayes_optimizer_kwargs['n_initial_points'] = \
#           checker.check_positive_integer(int(val_list[0]))
#         continue
#       if k in ('metric', 'base_metric'):
#         assert len(val_list) == 1
#         self.base_metric = val_list[0]
#         continue
# =======

      # Register key in common way
# >>>>>>> 2c2bb62db734310d5ab5fa0cb66e970e161ddebc
      self.register(k, *val_list)
      self.sys_flags.append(k)

  # endregion : Private Methods

# <<<<<<< HEAD
#   # region : BayesSearch
#
#   def _run_bayes(self, save=False, mark='', bayes_kwargs=None):
#     from tframe.utils.param_search.param_space import get_random_seed
#     if bayes_kwargs is None:
#       bayes_kwargs = dict()
#     assert isinstance(bayes_kwargs, dict)
#     random_state = bayes_kwargs.get('random_state', 1234)
#     search_initial_points = bayes_kwargs.get('search_initial_points', 2)
#     search_iters = bayes_kwargs.get('search_iters', 10)
#     search_iters = self.bayes_optimizer_kwargs.get('search_iters',
#                                                    search_iters)
#     if self.base_metric is None:
#       self.base_metric = bayes_kwargs.get('metric', 'best_loss')
#
#     optimizer_kwargs = {}
#     history = []
#
#     param_space = self.hyper_parameters
#     optimizer_kwargs['random_state'] = random_state
#     optimizer_kwargs['n_initial_points'] = search_initial_points
#
#     optimizer_kwargs.update(self.bayes_optimizer_kwargs)
#     optimizer_kwargs['random_state'] = get_random_seed(random_state)
#
#     optimizer = self._make_optimizer(param_space, optimizer_kwargs)
#
#     for param_id in range(search_iters):
#       history = self._step_bayes(optimizer, param_space, param_id,
#                                  search_iters, mark, save, history)
#
#   def _make_optimizer(self, param_space, optimizer_kwargs):
#     from tframe.utils.param_search.optimizer import Optimizer
#     from tframe.utils.param_search.param_space import param_space_2list
#     kwargs = optimizer_kwargs.copy()
#     kwargs['params_space'] = param_space_2list(param_space)
#     optimizer = Optimizer(**kwargs)
#     return optimizer
#
#   def _register_bayes(self, flag_name, parse, *val):
#     from tframe.utils.param_search.param_space import (FloatParamSpace,
#                                                        IntParamSpace,
#                                                        CategoricalParamSpace)
#
#     type = parse.get_kwarg('type', str)
#     # TODO: categorical
#     if type in ('categorical', 'c'):
#       self.hyper_parameters[flag_name] = \
#         CategoricalParamSpace(val[0][1:], transform='index')
#     elif type in ('int', 'integer', 'i'):
#       low = parse.get_kwarg('low', int)
#       high = parse.get_kwarg('high', int)
#       distribution = parse.get_kwarg('distribution', str, default='uniform')
#       self.hyper_parameters[flag_name] = IntParamSpace(low, high, distribution,
#                                                        transform='identity')
#     elif type in ('float', 'f'):
#       low = parse.get_kwarg('low', float)
#       high = parse.get_kwarg('high', float)
#       distribution = parse.get_kwarg('distribution', str, default='uniform')
#       self.hyper_parameters[flag_name] = FloatParamSpace(low, high,
#                                                          distribution,
#                                                          transform='identity')
#     else:
#       print('!! bayes param\'s type should be int, float, categorical, got {}'
#             .format(type))
#       return
#
#   def _step_bayes(self, optimizer, param_space, counter, iters, mark, save,
#                   history):
#     import pickle
#     from tframe.utils.param_search.param_space import point_2dict
#     from tframe import metrics as me
#     params = optimizer.ask()
#     # params = params[0]
#     params_dict = point_2dict(param_space, params[0])
#
#     # Grand self._add_script_suffix the highest priority
#     if self._add_script_suffix is not None: save = self._add_script_suffix
#     if save: self.common_parameters['script_suffix'] = '_{}{}'.format(
#       mark, counter)
#
#     try:
#       params_config = self._get_all_configs(params_dict)
#       self._apply_constraints(params_config)
#
#       params_list = self._get_config_strings(params_config)
#       params_string = ' '.join(params_list)
#       # if params_string in history: return
#       history.append(params_string)
#
#       console.show_status(
#         'Loading task ...', '[Run {}/{}]'.format(counter + 1, iters))
#       call([self._python_cmd, self.module_name] + params_list)
#       print()
#     except:
#       print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! get params error')
#       print("params_config {}".format(params_config))
#       print('params list {}'.format(params_list))
#       print('params string {}'.format(params_string))
#     # TODO: get results
#     # smaller is better
#     gather_summ_file = self.common_parameters.get('gather_summ_name',
#                                                   (self.default_summ_name +
#                                                    '.sum'))
#     try:
#       with open(gather_summ_file, 'rb') as f:
#         self.notes = pickle.load(f)
#       assert isinstance(self.notes, list)
#     except:
#       print('!! Failed to load {}'.format(gather_summ_file))
#       return
#
#     if self.base_metric not in self.notes[-1]._criteria:
#       print('!! metric {} not found in Result Notes {}'.
#             format(self.base_metric, self.notes[-1]._criteria))
#     results = - self.notes[-1]._criteria[self.base_metric]
#     # metric_quantity = me.get(self.base_metric)
#     # if metric_quantity.lower_is_better:
#     #   results = -results
#     optimizer.tell(params, [results])
#     return history
#
#   # endregion : BayesSearch
# =======
  # region: Search Engine Related

  def _get_summary(self):
    """This method knows the location of summary files."""
    # Get summary path
    summ_path = os.path.join(self.root_path, self.summ_file_name)
    # Load notes if exists
    if os.path.exists(summ_path):
      return [self._handle_hp_alias(n) for n in Note.load(summ_path)]
    else: return []

  @staticmethod
  def _handle_hp_alias(note):
    """TODO: this method is a compromise for HP alias, such as `lr`"""
    alias_dict = {'learning_rate': 'lr'}
    for name, alias in alias_dict.items():
      if name in note.configs:
        note.configs[alias] = note.configs.pop(name)
    return note

  def configure_engine(self, **kwargs):
    """This method is used for providing argument specifications in smart IDEs
       such as PyCharm"""
    kwargs.get('acq_kappa', None)
    kwargs.get('acq_n_points', None)
    kwargs.get('acq_n_restarts_optimizer', None)
    kwargs.get('acq_optimizer', None)
    kwargs.get('acq_xi', None)
    kwargs.get('acquisition', None)
    kwargs.get('add_script_suffix', None)
    kwargs.get('auto_set_hp_properties', None)
    kwargs.get('criterion', None)
    kwargs.get('expectation', None)
    kwargs.get('greater_is_better', None)
    kwargs.get('initial_point_generator', None)
    kwargs.get('n_initial_points', None)
    kwargs.get('prior', None)
    kwargs.get('strategy', None)
    kwargs.get('times', None)
    self.configure(**kwargs)

  def _export_log(self):
    # Determine filename
    log_path = os.path.join(
      self.root_path, '{}_log.txt'.format(self.pot.scroll.name))
    # Get log from scroll
    engine_logs = self.pot.scroll.get_log()
    # Create if not exist
    with safe_open(log_path, 'a'): pass
    # Write log at the beginning
    with safe_open(log_path, 'r+') as f:
      content = f.readlines()
      f.seek(0)
      f.truncate()
      f.write('{} summ: {}, scroll: {} \n'.format(
        get_time_string(), self.summ_file_name, self.pot.scroll.details))
      for line in engine_logs: f.write('  {}\n'.format(line))
      f.write('-' * 79 + '\n')
      f.writelines(content)

  def auto_set_hp_properties(self):
    """Set the properties of hyper-parameters automatically"""
    from tframe.alchemy.hyper_param import HyperParameter
    from tframe.configs.flag import Flag

    HubClass = type(self.shadow_th)
    for hp in self.pot.hyper_params:
      assert isinstance(hp, HyperParameter)
      # Find the corresponding flag in th
      flag = getattr(HubClass, hp.name)
      assert isinstance(flag, Flag)
      if hp.hp_type is None and flag.hp_type is not None:
        hp.hp_type = flag.hp_type
        console.show_status(
          '{}.hp_type set to {}'.format(hp.name, hp.hp_type), '[AutoSet]')
      if hp.scale is None and flag.hp_scale is not None:
        hp.scale = flag.hp_scale
        console.show_status(
          '{}.scale set to {}'.format(hp.name, hp.scale), '[AutoSet]')

  # endregion: Search Engine Related
# >>>>>>> 2c2bb62db734310d5ab5fa0cb66e970e161ddebc
