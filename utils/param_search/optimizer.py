import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.optimize import fmin_l_bfgs_b, OptimizeResult
from skopt.learning import (ExtraTreesRegressor, GaussianProcessRegressor,
                            GradientBoostingQuantileRegressor,
                            RandomForestRegressor)
from skopt.learning.gaussian_process.kernels import (ConstantKernel,
                                                     HammingKernel, Matern)
from skopt.acquisition import _gaussian_acquisition, gaussian_acquisition_1D

from tframe.utils.param_search.param_space import *


def is_regressor(estimator):
    return getattr(estimator, '_estimator_type', None) == 'regressor'


def cook_estimator(base_estimator, space=None, **kwargs):
    if isinstance(base_estimator, str):
        base_estimator = base_estimator.upper()
        allowed_estimators = ['GP', 'ET', 'RF', 'GBRT', 'DUMMY']
        if base_estimator not in allowed_estimators:
            raise ValueError('invalid estimator, should be in {}, got {}'
                             .format(allowed_estimators, base_estimator))
    elif not is_regressor(base_estimator):
        raise ValueError('base estimator should be a regressor, got {}'
                         .format(base_estimator))

    if base_estimator == 'GP':
        if space is not None:
            # space = Space(space)
            space = Space(normalize_param_space(space))
            n_params = space.transformed_n_params
            is_cat = space.is_categorical
        else:
            raise ValueError('expected a space instance, got None')
        cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
        if is_cat:
            other_kernel = HammingKernel(length_scale=np.ones(n_params))
        else:
            other_kernel = Matern(length_scale=np.ones(n_params),
                                  length_scale_bounds=[(0.01, 100)] * n_params,
                                  nu=2.5)
        base_estimator = GaussianProcessRegressor(
            kernel=cov_amplitude * other_kernel,
            normalize_y=True, noise='gaussian', n_restarts_optimizer=2
        )
    elif base_estimator == 'RF':
        base_estimator = RandomForestRegressor(n_estimators=100,
                                               min_samples_leaf=3)
    elif base_estimator == 'ET':
        base_estimator = ExtraTreesRegressor(n_estimators=100,
                                             min_samples_leaf=3)
    elif base_estimator == 'GRBT':
        grbt = GradientBoostingRegressor(n_estimators=30, loss='quantile')
        base_estimator = GradientBoostingQuantileRegressor(base_estimator=grbt)
    elif base_estimator == 'DUMMY':
        return None

    base_estimator.set_params(**kwargs)
    return base_estimator


def has_gradients(estimator):
    tree_estimator = (ExtraTreesRegressor, RandomForestRegressor,
                      GradientBoostingQuantileRegressor)
    if estimator is None:
        return False
    if isinstance(estimator, tree_estimator):
        return False
    categorical_gp = False
    if hasattr(estimator, 'kernel'):
        params = estimator.get_params()
        categorical_gp = (isinstance(estimator.kernel, HammingKernel) or
                          any([isinstance(params[p], HammingKernel)
                               for p in params]))
    return not categorical_gp


class Optimizer(object):
    def __init__(
            self, params_space, base_estimator='gp', n_initial_points=5,
            acq_func='gp_hedge', acq_optimizer='auto', random_state=None,
            model_queue_size=None, acq_func_kwargs=None,
            acq_optimizer_kwargs=None
    ):
        self.rs = get_random_seed(random_state)

        allowed_acq_funcs = ['gp_hedge', 'EI', 'LCB', 'PI', 'EIps', 'PIps']
        if acq_func not in allowed_acq_funcs:
            raise ValueError('acq func should in {}, got {}'
                             .format(allowed_acq_funcs, acq_func))
        self.acq_func = acq_func
        self.acq_func_kwargs = acq_func_kwargs
        if acq_func == 'gp_hedge':
            self.cand_acq_funcs = ['EI', 'LCB', 'PI']
            self.gains = np.zeros(3)
        else:
            self.cand_acq_funcs = [self.acq_func]
        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.eta = acq_func_kwargs.get('eta', 1.0)

        if n_initial_points < 0:
            raise ValueError('n_initial_points should be larger than 0, got {}'
                             .format(n_initial_points))
        self.n_initial_points = n_initial_points

        if isinstance(base_estimator, str):
            base_estimator = cook_estimator(
                base_estimator, space=params_space,
                random_state=self.rs.randint(0, np.iinfo(np.int32).max)
            )
        if not is_regressor(base_estimator) and base_estimator is not None:
            raise ValueError('base estimator should be regressor, got {}'
                             .format(base_estimator))
        is_multi_regressor = isinstance(base_estimator, MultiOutputRegressor)
        if 'ps' in self.acq_func and not is_multi_regressor:
            self.base_estimator = MultiOutputRegressor(base_estimator)
        else:
            self.base_estimator = base_estimator

        if acq_optimizer == 'auto':
            if has_gradients(self.base_estimator):
                acq_optimizer = 'lbfgs'
            else:
                acq_optimizer = 'sampling'
        if acq_optimizer not in ['lbfgs', 'sampling']:
            raise ValueError('acq_optimizer should be lbfgs or sampling, got '
                             '{}'.format(acq_optimizer))
        if (not has_gradients(self.base_estimator) and
                acq_optimizer != 'sampling'):
            raise ValueError('the regressor {} should run with '
                             'acq_optimizer=sampling'.format(base_estimator))
        self.acq_optimizer = acq_optimizer

        if acq_optimizer_kwargs is None:
            acq_optimizer_kwargs = {}
        self.n_points = acq_optimizer_kwargs.get('n_points', 100)
        self.n_restarts_optimizer = acq_optimizer_kwargs.get(
            'n_restarts_optimizer', 5)
        self.n_jobs = acq_optimizer_kwargs.get('n_jobs', 1)
        self.acq_optimizer_kwargs = acq_optimizer_kwargs

        if isinstance(self.base_estimator, GaussianProcessRegressor):
            params_space = normalize_param_space(params_space)
        self.space = Space(params_space)

        self._cat_inds = []
        self._non_cat_inds = []
        for ind, ps in enumerate(self.space.param_space):
            if isinstance(ps, CategoricalParamSpace):
                self._cat_inds.append(ind)
            else:
                self._non_cat_inds.append(ind)

        if not isinstance(model_queue_size, (int, type(None))):
            raise TypeError('model_queue_size should be int or None, got {}'
                            .format(model_queue_size))
        self.max_model_queue_size = model_queue_size
        self.models = []
        self.xi = []
        self.yi = []
        self.cache = {}

    def _ask(self):
        # return a list which contains several param points
        if self.n_initial_points > 0 or self.base_estimator is None:
            # return self.space.rvs(random_state=self.rs)[0]
            return self.space.rvs(random_state=self.rs)
        else:
            if not self.models:
                raise RuntimeError('random evaluations exhausted and no more '
                                   'models have been fit')
            next_x = self._next_x
            min_delta_x = min([self.space.distance(next_x, xi)
                               for xi in self.xi])
            if abs(min_delta_x) <= 1e-8:
                print('this candidate has been evaluated before')
            return [next_x]

    def ask(self, n_points=None):
        if n_points is None:
            return self._ask()
        # TODO: if n_points > 1
        pass

    def _check_y_is_valid(self, x, y):
        if 'ps' in self.acq_func:
            # TODO: ps
            pass
        elif is_listlike(y) and is_2d_listlike(x):
            for yy in y:
                if not isinstance(yy, (int, float, np.float32)):
                    raise ValueError('expected y to be a list of scalars, got'
                                     '{}, type {}'.format(y, type(yy)))
        elif is_listlike(x):
            if not isinstance(y, (int, float)):
                raise ValueError('expected y to be a scalar, got {}'.format(y))
        else:
            raise ValueError('x {} and y {} are not compatible'.format(x, y))

    def _tell(self, x, y, fit=True):
        if 'ps' in self.acq_func:
            # TODO: ps
            pass
        elif is_listlike(y) and is_2d_listlike(x):
            self.xi.extend(x)
            self.yi.extend(y)
            self.n_initial_points -= len(y)
        elif is_listlike(x):
            self.xi.append(x)
            self.yi.append(y)
            self.n_initial_points -= 1
        else:
            raise ValueError('x {} and y {} are not compatible'.format(x, y))

        self.cache = {}
        if (fit and self.n_initial_points <= 0 and
                self.base_estimator is not None):
            transformed_bounds = np.array(self.space.transformed_bounds)
            estimator = self.base_estimator
            estimator.fit(self.space.transform(self.xi), self.yi)

            if hasattr(self, 'next_xs_') and self.acq_func == 'gp_hedge':
                self.gains -= estimator.predict(np.vstack(self.next_xs_))

            if self.max_model_queue_size is None:
                self.models.append(estimator)
            elif len(self.models) < self.max_model_queue_size:
                self.models.append(estimator)
            else:
                self.models.pop(0)
                self.models.append(estimator)

            x = self.space.transform(self.space.rvs(n_samples=self.n_points,
                                                    random_state=self.rs))
            self.next_xs_ = []
            for cand_acq_func in self.cand_acq_funcs:
                values = _gaussian_acquisition(
                    X=x, model=estimator, y_opt=np.min(self.yi),
                    acq_func=cand_acq_func,
                    acq_func_kwargs=self.acq_func_kwargs
                )
                if self.acq_optimizer == 'sampling':
                    next_x = x[np.argmin(values)]
                elif self.acq_optimizer == 'lbfgs':
                    x0 = np.asarray(x)[
                        np.argsort(values)[:self.n_restarts_optimizer]
                    ]
                    res = [fmin_l_bfgs_b(gaussian_acquisition_1D, xx,
                                         args=(estimator, np.min(self.yi),
                                               cand_acq_func,
                                               self.acq_func_kwargs),
                                         bounds=self.space.transformed_bounds,
                                         approx_grad=False,
                                         maxiter=20) for xx in x0]
                    cand_xs = np.array([r[0] for r in res])
                    cand_acqs = np.array([r[1] for r in res])
                    next_x = cand_xs[np.argmin(cand_acqs)]

                if not self.space.is_categorical:
                    next_x = np.clip(next_x, transformed_bounds[:, 0],
                                     transformed_bounds[:, 1])
                self.next_xs_.append(next_x)

            if self.acq_func == 'gp_hedge':
                logits = np.array(self.gains)
                logits -= np.max(logits)
                exp_logits = np.exp(self.eta * logits)
                probs = exp_logits / np.sum(exp_logits)
                next_x = self.next_xs_[np.argmax(self.rs.multinomial(1, probs))]
            else:
                next_x = self.next_xs_[0]
            self._next_x = self.space.inverse_transform(
                next_x.reshape((1, -1)))[0]

        return create_result(self.xi, self.yi, self.space, self.rs,
                             models=self.models)

    def tell(self, x, y, fit=True):
        check_x_in_space(x, self.space)
        self._check_y_is_valid(x, y)

        # TODO: if acq_func has ps
        if 'ps' in self.acq_func:
            pass
        return self._tell(x, y, fit=fit)


def create_result(xi, yi, space=None, rs=None, specs=None, models=None):

    res = OptimizeResult()
    yi = np.asarray(yi)
    if np.ndim(yi) == 2:
        res.log_time = np.ravel(yi[:, 1])
        yi = np.ravel(yi[:, 0])
    best = np.argmin(yi)
    res.x = xi[best]
    res.fun = yi[best]
    res.func_vals = yi
    res.x_iters = xi
    res.models = models
    res.space = space
    res.random_state = rs
    res.specs = specs
    return res
