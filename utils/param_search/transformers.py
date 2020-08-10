import numpy as np
from sklearn.preprocessing import LabelBinarizer


class Transformer(object):
    def fit(self, x):
        return self

    def transform(self, x):
        raise NotImplementedError

    def inverse_transform(self, x_t):
        raise NotImplementedError


class Identity(Transformer):
    def transform(self, x):
        return x

    def inverse_transform(self, x_t):
        return x_t


class Normalize(Transformer):
    def __init__(self, low, high, is_int=False):
        self.low = float(low)
        self.high = float(high)
        self.is_int = is_int

    def transform(self, x):
        x = np.asarray(x)
        if self.is_int:
            if np.any(np.round(x) > self.high):
                raise ValueError('the upper bound is {}, got {} values higher '
                                 'than it'.format(self.high, x))
            if np.any(np.round(x) < self.low):
                raise ValueError('the lower bound is {}, got {} values less '
                                 'than it'.format(self.low, x))
        else:
            if np.any(x > self.high + 1e-8):
                raise ValueError('the upper bound is {}, got {} values higher '
                                 'than it'.format(self.high, x))
            if np.any(x < self.low - 1e-8):
                raise ValueError('the lower bound is {}, got {} values less '
                                 'than it'.format(self.low, x))

        if self.is_int:
            return ((np.round(x).astype(np.int) - self.low) /
                    (self.high - self.low))
        else:
            return (x - self.low) / (self.high - self.low)

    def inverse_transform(self, x_t):
        x_t = np.asarray(x_t)
        if np.any(x_t > 1.0):
            raise ValueError('all values should be less than 1.0, got {}'
                             .format(x_t))
        if np.any(x_t < 0.0):
            raise ValueError('all values should be larger than 0.0, got {}'
                             .format(x_t))
        x_o = x_t * (self.high - self.low) + self.low
        if self.is_int:
            return [int(np.round(xx)) for xx in x_o]
            # return np.round(x_o).astype(int)
        return x_o


class LogN(Transformer):
    def __init__(self, base):
        self._base = base

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        return np.log10(x) / np.log10(self._base)

    def inverse_transform(self, x_t):
        x_t = np.asarray(x_t, dtype=float)
        return self._base ** x_t


class CategoricalEncoder(Transformer):
    def __init__(self):
        self._lb = LabelBinarizer()

    def fit(self, x):
        self.mapping = {v: i for i, v in enumerate(x)}
        self.inverse_mapping = {i: v for i, v in enumerate(x)}
        self._lb.fit([self.mapping[v] for v in x])
        self.n_classes = len(self._lb.classes_)
        return self

    def transform(self, x):
        x = np.asarray(x)
        return self._lb.transform([self.mapping[v] for v in x])

    def inverse_transform(self, x_t):
        x_t = np.asarray(x_t)
        return [self.inverse_mapping[i] for i in
                self._lb.inverse_transform(x_t)]


class StringEncoder(Transformer):
    def __init__(self, dtype=str):
        self.dtype = dtype

    def fit(self, x):
        if len(x) > 0:
            self.dtype = type(x[0])
            for xx in x:
                if self.dtype != type(xx):
                    raise ValueError('StringEncoder should be fit on a list of'
                                     'same type, got {}'.format(x))
        self.mapping = {str(v): v for v in x}
        return self

    def transform(self, x):
        # TODO: how to str a list etc, can only str() ?
        return [str(xx) for xx in x]

    def inverse_transform(self, x_t):
        return [self.mapping[xx] for xx in x_t]


class IndexEncoder(Transformer):
    def __init__(self, dtype=int):
        self.dtype = dtype

    def fit(self, x):
        self.mapping = {i: v for i, v in enumerate(x)}
        self.inverse_mapping = {v: i for i, v in enumerate(x)}
        return self

    def transform(self, x):
        return [self.inverse_mapping[xx] for xx in x]

    def inverse_transform(self, x_t):
        return [self.mapping[xx] for xx in x_t]


class Pipeline(Transformer):
    def __init__(self, transformers):
        self.transformers = list(transformers)
        for transformer in self.transformers:
            if not isinstance(transformer, Transformer):
                raise ValueError('expected Transformer, got {}'
                                 .format(transformer))

    def fit(self, x):
        for transformer in self.transformers:
            transformer.fit(x)
        return self

    def transform(self, x):
        for transformer in self.transformers:
            x = transformer.transform(x)
        return x

    def inverse_transform(self, x_t):
        for transformer in self.transformers[::-1]:
            x_t = transformer.inverse_transform(x_t)
        return x_t
