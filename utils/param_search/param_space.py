import numpy as np
from scipy.stats.distributions import uniform, randint, rv_discrete

from acorn.utils.param_search.transformers import *
from acorn.data.data_utils import get_random_seed


def _uniform_inclusive(loc=0.0, scale=1.0):
    return uniform(loc=loc, scale=np.nextafter(scale, scale + 1.))


def check_param_space(param_space, transform=None):
    if isinstance(param_space, BaseParamSpace):
        return param_space
    if not isinstance(param_space, (list, tuple, np.ndarray)):
        raise ValueError('param_space should be a list or a tuple, got {} with'
                         'type {}'.format(param_space, type(param_space)))
    if len(param_space) == 1:
        return CategoricalParamSpace(param_space, transform=transform)
    if len(param_space) == 2:
        if any([isinstance(ps, (str, bool, np.bool_)) for ps in param_space]):
            return CategoricalParamSpace(param_space, transform=transform)
        elif all([isinstance(ps, int) for ps in param_space]):
            return IntParamSpace(*param_space, transform=transform)
        elif any([isinstance(ps, (float, int)) for ps in param_space]):
            return FloatParamSpace(*param_space, transform=transform)
        else:
            raise ValueError('invalid param space, got {}'.format(param_space))
    if len(param_space) == 3:
        if (all([isinstance(ps, int) for ps in param_space[:2]]) and
                param_space[2] in ['uniform', 'log-uniform']):
            return IntParamSpace(*param_space, transform=transform)
        elif (any([isinstance(ps, (float, int)) for ps in param_space[:2]]) and
              param_space[2] in ['uniform', 'log-uniform']):
            return FloatParamSpace(*param_space, transform=transform)
        else:
            return CategoricalParamSpace(param_space, transform=transform)
    if len(param_space) == 4:
        if (all([isinstance(ps, int) for ps in param_space[:2]]) and
                param_space[2] == 'log-uniform' and
                isinstance(param_space[3], int)):
            return IntParamSpace(*param_space, transform=transform)
        elif (any([isinstance(ps, (float, int)) for ps in param_space[:2]]) and
              param_space[2] == 'log-uniform' and
              isinstance(param_space[3], int)):
            return FloatParamSpace(*param_space, transform=transform)
    if len(param_space) > 3:
        return CategoricalParamSpace(param_space, transform=transform)
    raise ValueError('invalid param space, got {}'.format(param_space))


def check_param_spaces(param_spaces):
    if len(param_spaces) == 0:
        raise ValueError('the param spaces can not be empty, got {}'
                         .format(param_spaces))
    if isinstance(param_spaces, dict):
        param_spaces = [param_spaces]
    if isinstance(param_spaces, list):
        dicts_only = []
        for ps in param_spaces:
            if isinstance(ps, tuple):
                if len(ps) != 2:
                    raise ValueError('tuple in list of param_spaces should'
                                     'have length of 2, and contain (dict,'
                                     ' int) got {}'.format(ps))
                subspace, n_iter = ps
                if (not isinstance(n_iter, int)) or n_iter < 0:
                    raise ValueError('number of iteration should be a '
                                     'integer and larger than 0, got {}'
                                     .format(n_iter))
                dicts_only.append(subspace)
            elif isinstance(ps, dict):
                dicts_only.append(ps)
            else:
                raise TypeError('param spaces should be a dict or a tuple'
                                'of (dict, int), got {}'.format(ps))

        for sp in dicts_only:
            for k, v in sp.items():
                check_param_space(v)
    else:
        raise ValueError('param spaces should be dict or list of dict, got'
                         '{}'.format(param_spaces))


def is_listlike(x):
    return isinstance(x, (list, tuple))


def is_2d_listlike(x):
    return np.all([is_listlike(xx) for xx in x])


def check_x_in_space(x, param_space):
    assert isinstance(param_space, Space)
    if is_2d_listlike(x):
        x = x
    elif is_listlike(x):
        x = [x]
    if not np.all([p in param_space for p in x]):
        raise ValueError('not all points are in the bounds of the param '
                         'space, got {}'.format(x))
    if any([len(p) != len(param_space.param_space) for p in x]):
        raise ValueError('not all points have the same length as the '
                         'space, got {}'.format(x))


def param_space_2list(param_space):
    param_space_list = [param_space[k] for k in sorted(param_space.keys())]
    return param_space_list


def point_2dict(param_space, param_list):
    params_dict = {}
    for k, v in zip(sorted(param_space.keys()), param_list):
        params_dict[k] = v
    return params_dict


class BaseParamSpace(object):

    @property
    def transformed_size(self):
        return 1

    @property
    def transformed_bounds(self):
        raise NotImplementedError

    def transform(self, x):
        return self.transformer.transform(x)

    def inverse_transform(self, x_t):
        return self.transformer.inverse_transform(x_t)

    def rvs(self, n_samples=1, random_state=None):
        rs = get_random_seed(random_state)
        samples = self._rv.rvs(size=n_samples, random_state=rs)
        return self.inverse_transform(samples)

    def distance(self, a, b):
        raise NotImplementedError


class FloatParamSpace(BaseParamSpace):
    def __init__(self, low, high, distribution='uniform', base=10,
                 transform=None, name=None, dtype=float):
        if high <= low:
            raise ValueError('the lower bound should be less than the upper '
                             'bound, got low {} high {}'.format(low, high))
        self.low = low
        self.high = high

        if distribution not in ['uniform', 'log-uniform']:
            raise ValueError("distribution should be 'uniform' or "
                             "'log-uniform, got {}".format(distribution))
        self.distribution = distribution
        self.base = base
        self.log_base = np.log10(base)

        if transform is None:
            transform = 'identity'
        self.transform_ = transform
        if transform == 'normalize':
            # TODO: random variable search method
            self._rv = _uniform_inclusive(0., 1.)
            if self.distribution == 'uniform':
                self.transformer = Pipeline([Identity(), Normalize(low, high)])
            else:
                self.transformer = Pipeline(
                    [LogN(base),
                     Normalize(np.log10(low) / self.log_base,
                               np.log10(high) / self.log_base)]
                )
        elif transform == 'identity':
            # TODO:
            if self.distribution == 'uniform':
                self._rv = _uniform_inclusive(low, high - low)
                self.transformer = Identity()
            else:
                self._rv = _uniform_inclusive(
                    np.log10(low) / self.log_base,
                    (np.log10(high) - np.log10(low)) / self.log_base)
                self.transformer = LogN(base)
        else:
            raise ValueError("transform should be 'identity' or 'normalize', "
                             "got {}".format(transform))

        if (isinstance(dtype, str) and dtype not in
                ['float', 'float16', 'float32', 'float64']):
            raise TypeError('dtype should be float, got {}'.format(dtype))
        elif (isinstance(dtype, type) and dtype not in
              [float, np.float, np.float16, np.float32, np.float64]):
            raise TypeError('dtype should be float, got {}'.format(dtype))
        self.dtype = dtype
        self.name = name

    def __eq__(self, other):
        return (type(self) is type(other) and
                np.allclose([self.low], [other.low]) and
                np.allclose([self.high], [other.high]))

    def __contains__(self, item):
        if isinstance(item, list):
            item = np.array(item)
        return self.low <= item <= self.high

    @property
    def bound(self):
        return self.low, self.high

    @property
    def transformed_bounds(self):
        if self.transform_ == 'normalize':
            return 0.0, 1.0
        else:
            if self.distribution == 'uniform':
                return self.low, self.high
            else:
                return np.log10(self.low), np.log10(self.high)

    def distance(self, a, b):
        return abs(a - b)


class IntParamSpace(BaseParamSpace):
    def __init__(self, low, high, distribution='uniform', base=10,
                 transform=None, name=None, dtype=int):
        if high <= low:
            raise ValueError('the lower bound should be less than the upper '
                             'bound, got low {} high {}'.format(low, high))
        self.low = low
        self.high = high

        if distribution not in ['uniform', 'log-uniform']:
            raise ValueError("distribution should be 'uniform' or "
                             "'log-uniform, got {}".format(distribution))
        self.distribution = distribution
        self.base = base
        self.log_base = np.log10(base)

        if transform is None:
            transform = 'identity'
        self.transform_ = transform
        if transform == 'normalize':
            # TODO:
            self._rv = _uniform_inclusive(0.0, 1.0)
            if self.distribution == 'uniform':
                self.transformer = Pipeline(
                    [Identity(), Normalize(low, high, is_int=True)]
                )
            else:
                self.transformer = Pipeline(
                    [LogN(self.base),
                     Normalize(np.log10(low) / self.log_base,
                               np.log10(high) / self.log_base, is_int=True)]
                )
        elif transform == 'identity':
            # TODO:
            if self.distribution == 'uniform':
                self._rv = randint(self.low, self.high + 1)
                self.transformer = Identity()
            else:
                self._rv = _uniform_inclusive(
                    np.log10(low) / self.log_base,
                    (np.log10(high) - np.log10(low)) / self.log_base
                )
                self.transformer = LogN(self.base)
        else:
            raise ValueError("transform should be 'identity' or 'normalize', "
                             "got {}".format(transform))

        if (isinstance(dtype, str) and dtype not in
                ['int', 'int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32',
                 'uint64']):
            raise TypeError('dtype should be int, got {}'.format(dtype))
        elif (isinstance(dtype, type) and dtype not in
              [int, np.int, np.int8, np.int16, np.int32, np.int64, np.uint8,
               np.uint16, np.uint32, np.uint64]):
            raise TypeError('dtype should be int, got {}'.format(dtype))
        self.dtype = dtype
        self.name = name

    def __eq__(self, other):
        return (type(self) is type(other) and
                np.allclose([self.low], [other.low]) and
                np.allclose([self.high], [other.high]))

    def __contains__(self, item):
        if isinstance(item, list):
            item = np.array(item)
        return self.low <= item <= self.high

    @property
    def bound(self):
        return self.low, self.high

    @property
    def transformed_bounds(self):
        if self.transform_ == 'normalize':
            return 0, 1
        else:
            if self.distribution == 'uniform':
                return self.low, self.high
            else:
                return np.log10(self.low), np.log10(self.high)

    def inverse_transform(self, x_t):
        inv_transform = super(IntParamSpace, self).inverse_transform(x_t)
        if isinstance(inv_transform, list):
            inv_transform = [int(np.round(xx)) for xx in inv_transform]
            # inv_transform = np.array(inv_transform, dtype=self.dtype)
        # return np.round(inv_transform).astype(self.dtype)
        return inv_transform

    def distance(self, a, b):
        return abs(a - b)


class CategoricalParamSpace(BaseParamSpace):
    def __init__(self, categories, distribution=None, transform=None, name=None):
        self.categories = categories
        self.name = name

        if transform is None:
            transform = 'index'
        self.transform_ = transform
        if transform == 'identity':
            self.transformer = Identity()
            self.transformer.fit(categories)
        elif transform == 'onehot':
            self.transformer = CategoricalEncoder()
            self.transformer.fit(categories)
        elif transform == 'string':
            self.transformer = StringEncoder()
            self.transformer.fit(categories)
        elif transform == 'index':
            self.transformer = IndexEncoder()
            self.transformer.fit(categories)
        else:
            raise ValueError("transform should be 'identity', 'onehot', "
                             "'index', or 'string', got {}".format(transform))

        if distribution is None:
            self.distribution = np.tile(1. / len(categories), len(categories))
        else:
            if not isinstance(distribution, (list, np.ndarray)):
                raise TypeError('distribution should be a list or a 1-D array,'
                                ' got {} with type {}'
                                .format(distribution, type(distribution)))
            if np.sum(distribution) != 1:
                raise ValueError('the sum of distribution should be 1, got {} '
                                 'with sum {}'.format(distribution,
                                                      np.sum(distribution)))
            self.distribution = distribution

        # TODO: get random_variable method
        self._rv = rv_discrete(values=(range(len(categories)),
                                       self.distribution))

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.categories == other.categories and
                np.allclose(self.distribution, other.distribution))

    def __contains__(self, item):
        return item in self.categories

    @property
    def bounds(self):
        return self.categories

    @property
    def transformed_size(self):
        # TODO: ???
        if self.transform_ == 'onehot':
            size = len(self.categories)
            return size if size != 2 else 1
        return 1

    @property
    def transformed_bounds(self):
        if self.transformed_size == 1:
            return 0.0, 1.0
        else:
            return [(0.0, 1.0) for i in range(self.transformed_size)]

    def rvs(self, n_samples=1, random_state=None):
        choices = self._rv.rvs(size=n_samples, random_state=random_state)
        if isinstance(choices, int):
            return self.categories[choices]
        else:
            return [self.categories[c] for c in choices]

    def distance(self, a, b):
        return 1 if a != b else 0


def normalize_param_space(param_space):
    space = Space(param_space)
    transformed_params_space = []
    if space.is_categorical:
        for ps in space:
            transformed_params_space.append(
                CategoricalParamSpace(ps.categories, ps.distribution,
                                      name=ps.name, transform='string')
            )
    else:
        for ps in space.param_space:
            if isinstance(ps, CategoricalParamSpace):
                transformed_params_space.append(ps)
            elif isinstance(ps, FloatParamSpace):
                transformed_params_space.append(
                    FloatParamSpace(ps.low, ps.high, ps.distribution,
                                    name=ps.name, transform='normalize',
                                    dtype=ps.dtype)
                )
            elif isinstance(ps, IntParamSpace):
                transformed_params_space.append(
                    IntParamSpace(ps.low, ps.high, ps.distribution,
                                  name=ps.name, transform='normalize',
                                  dtype=ps.dtype)
                )
            else:
                raise ValueError('unknown param space type, got {} with type '
                                 '{}'.format(ps, type(ps)))
    return Space(transformed_params_space)


class Space(object):
    def __init__(self, param_space):
        self.param_space = [check_param_space(ps) for ps in param_space]

    def __eq__(self, other):
        return all([a == b for a, b in
                    zip(self.param_space, other.param_space)])

    def __iter__(self):
        return iter(self.param_space)

    def __contains__(self, item):
        for it, ps in zip(item, self.param_space):
            if it not in ps:
                return False
        return True

    @property
    def n_params(self):
        return len(self.param_space)

    @property
    def transformed_n_params(self):
        return sum([ps.transformed_size for ps in self.param_space])

    @property
    def transformed_bounds(self):
        b = []
        for ps in self.param_space:
            if ps.transformed_size == 1:
                b.append(ps.transformed_bounds)
            else:
                b.extend(ps.transformed_bounds)
        return b

    @property
    def is_categorical(self):
        return all([isinstance(ps, CategoricalParamSpace)
                    for ps in self.param_space])

    def transform(self, x):
        columns = []
        for ps in self.param_space:
            columns.append([])
        for i in range(len(x)):
            for j in range(self.n_params):
                columns[j].append(x[i][j])
        for j in range(self.n_params):
            columns[j] = self.param_space[j].transform(columns[j])

        # x_t = np.hstack([np.asarray(c).reshape((len(x), -1)) for c in columns])
        x_t = [[xx[j] for xx in columns] for j in range(len(x))]
        return x_t

    def inverse_transform(self, x_t):
        columns = []
        start = 0
        for j in range(self.n_params):
            ps = self.param_space[j]
            offset = ps.transformed_size
            if offset == 1:
                columns.append(ps.inverse_transform(x_t[:, start]))
            else:
                columns.append(ps.inverse_transform(x_t[:,
                                                    start: start + offset]))
            start += offset
        rows = []
        for i in range(len(x_t)):
            r = []
            for j in range(self.n_params):
                r.append(columns[j][i])
            rows.append(r)
        return rows

    def rvs(self, n_samples=1, random_state=None):
        rs = get_random_seed(random_state)
        columns = []
        for ps in self.param_space:
            columns.append(ps.rvs(n_samples=n_samples, random_state=rs))
        rows = []
        for i in range(n_samples):
            r = []
            for j in range(self.n_params):
                r.append(columns[j][i])
            rows.append(r)
        return rows

    def distance(self, a, b):
        distance = 0
        for aa, bb, ps in zip(a, b, self.param_space):
            distance += ps.distance(aa, bb)
        return distance


if __name__ == '__main__':

    def aaa():
        param_space = {
            'groups': CategoricalParamSpace([2, 4, 8, 16], transform='identity'),
            'init_filters': CategoricalParamSpace([16, 32, 64], transform='identity')
        }
        param_space = param_space_2list(param_space)
        param_space = normalize_param_space(param_space)
        space = Space(param_space)
        a = [[2, 4, 4, 8], [16, 16, 64]]
        space.transform(a)

    aaa()

    def bbb():
        xk = np.arange(7)
        pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)
        custm = rv_discrete(name='aaa', values=(xk, pk))

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        ax.plot(xk, custm.pmf(xk), 'ro', ms=12, mec='r')
        ax.vlines(xk, 0, custm.pmf(xk), colors='r', lw=4)
        plt.show()