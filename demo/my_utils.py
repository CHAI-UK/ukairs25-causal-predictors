"""
This code has been extracted from:
https://github.com/francescomontagna/causally
"""

import random

import numpy as np
import networkx as nx

from typing import Union
from abc import ABCMeta, abstractmethod

from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.metrics import accuracy_score

from pickle import load

def load_model(filename):
    with open(f"{filename}.pkl", "rb") as f:
        m = load(f)
    return m

def load_models(case, conf, noise, scm, m_type, n):
    fname = f"./model_{case}_{conf}_{scm}_{noise}_{m_type}_{n}"
    models = {}
    for n in ['all', 'causal']:
        models[n] = load_model(f"{fname}_{n}")
    return models

def get_labels(X):
    # first column is the label
    # trainsform into a binary class
    return (X[:, 0] > 0).astype(float)

def get_features(X):
    return X[:, 1:]

def init_features(case, conf):
    features = {}
    if case == '3-conf':
        if conf == 'hidden':
            features['causal'] = [0]
            features['all'] = [0, 1, 2]
        elif conf == 'observed':
            features['causal'] = [0, 3, 4, 5]
            features['all'] = list(range(6))
    elif case in ['TC-conf', 'TAC-conf', 'TS-conf', 'SC-conf']:
        if conf == 'hidden':
            features['causal'] = [0]
            features['all'] = [0, 1, 2]
        elif conf == 'observed':
            features['causal'] = [0, 3]
            features['all'] = [0, 1, 2, 3]
    
    return features

def set_random_seed(seed=1):
    np.random.seed(seed)
    random.seed(seed)

def sample(n, adj, mechanism, noise_gen, shift_interventions={}):
    set_random_seed()
    
    # Sample the noise
    noise = noise_gen.sample((n, len(adj)))
    X = noise.copy()

    # Generate the data starting from source nodes
    graph_order = topological_order(adj)
    for node in graph_order:
        parents = np.nonzero(adj[:, node])[0]
        if len(parents) > 0:
            # assuming additive noise models
            X[:, node] = mechanism.predict(X[:, parents]) + noise[:, node]
        
        # assuming the intervention is additive
        if node in shift_interventions:
            if shift_interventions[node] != 0:
                X[:, node] += shift_interventions[node] # soft intervention

    return X

def define_adjmat(case):
    if case == '3-conf':
        adj = np.array([[0,0,1,0,0,0,0], # target
                        [1,0,0,1,0,0,0], # causal feature
                        [0,0,0,0,0,0,0], # anti-causal feature
                        [0,0,0,0,0,0,0], # spurious feature
                        [1,1,0,0,0,0,0], # potential confounder of target and causal feature
                        [1,0,1,0,0,0,0], # potential confounder of target and anti-causal feature
                        [1,0,0,1,0,0,0]  # potential confounder of target and spurios feature
                        ])
        cols = ['target', 'causal', 'anti-causal', 'spurious', 'TC-conf', 'TAC-conf', 'TS-conf']
    elif case == 'TC-conf':
        adj = np.array([[0,0,1,0,0], # target
                        [1,0,0,1,0], # causal feature
                        [0,0,0,0,0], # anti-causal feature
                        [0,0,0,0,0], # spurious feature
                        [1,1,0,0,0]  # potential confounder of target and causal feature
                        ])
        cols = ['target', 'causal', 'anti-causal', 'spurious', 'TC-conf']
    elif case == 'TAC-conf':
        adj = np.array([[0,0,1,0,0], # target
                        [1,0,0,1,0], # causal feature
                        [0,0,0,0,0], # anti-causal feature
                        [0,0,0,0,0], # spurious feature
                        [1,0,1,0,0]  # potential confounder of target and anti-causal feature
                        ])
        cols = ['target', 'causal', 'anti-causal', 'spurious', 'TAC-conf']
    elif case == 'TS-conf':
        adj = np.array([[0,0,1,0,0], # target
                        [1,0,0,1,0], # causal feature
                        [0,0,0,0,0], # anti-causal feature
                        [0,0,0,0,0], # spurious feature
                        [1,0,0,1,0]  # potential confounder of target and spurios feature
                        ])
        cols = ['target', 'causal', 'anti-causal', 'spurious', 'TS-conf']
    elif case == 'SC-conf':
        adj = np.array([[0,0,1,0,0], # target
                        [1,0,0,1,0], # causal feature
                        [0,0,0,0,0], # anti-causal feature
                        [0,0,0,0,0], # spurious feature
                        [0,1,0,1,0]  # potential confounder of spurious and causal feature
                        ])
        cols = ['target', 'causal', 'anti-causal', 'spurious', 'SC-conf']
    else:
        raise ValueError('Incorrect graph case')
    
    return adj, cols

def define_setup(case, noise, scm):
    adjacency, columns = define_adjmat(case)

    if noise == 'normal':
        noise = Normal()
    else:
        raise ValueError('Incorrect noise value')
    
    if scm == 'linear':
        mechanism = LinearMechanism()
    else:
        raise ValueError('Incorrect SCM type')

    return adjacency, noise, mechanism, columns

def eval_models(X, y, models, features):
    results = {}
    for name in models:
        X_feats = X[:, features[name]]
        if len(features[name]) == 1:
            X_feats = X_feats.reshape(-1, 1)

        y_hat = models[name].predict(X_feats)
        results[name] = accuracy_score(y, y_hat)
    
    return results

# *** Abstract base classes *** #
class Distribution(metaclass=ABCMeta):
    """Base class to represent noise distributions."""

    @abstractmethod
    def sample(self, size: tuple[int]) -> np.array:
        raise NotImplementedError
    
# *** Wrappers of numpy distributions *** #
class Normal(Distribution):
    """Wrapper for ``numpy.random.normal()`` sampler.

    Parameters
    ----------
    loc: Union[float, np.array of floats], default 0
        The mean of the sample.
    std: Union[float, np.array of floats], default 1
        The standard deviation of the sample.
    """

    def __init__(
        self, loc: Union[float, np.array] = 0.0, std: Union[float, np.array] = 1.0
    ):
        super().__init__()
        self.loc = loc
        self.std = std

    def sample(self, size: tuple[int]) -> np.array:
        """Draw random samples from a Gaussian distribution.

        Parameters
        ----------
        size: tuple[int]
            Required shape of the random sample.
        """
        if len(size) != 2:
            ValueError(
                f"Expected number of input dimensions is 2, but were given {len(size)}."
            )
        return np.random.normal(self.loc, self.std, size)


class Exponential(Distribution):
    r"""Wrapper for ``numpy.random.exponential()`` sampler.

    Parameters
    ----------
    scale: Union[float, np.array of floats], default 1
        The scale parameter :math:`\beta = \frac{1}{\lambda}`, must be non-negative.
    """

    def __init__(self, scale: Union[float, np.array] = 1.0):
        super().__init__()
        self.scale = scale

    def sample(self, size: tuple[int]) -> np.array:
        """Draw random samples from an exponential distribution.

        Parameters
        ----------
        size: tuple[int]
            Required shape of the random sample.
        """
        if len(size) != 2:
            ValueError(
                f"Expected number of input dimensions is 2, but were given {len(size)}."
            )
        return np.random.exponential(self.scale, size)


class Uniform(Distribution):
    r"""Wrapper for ``numpy.random.uniform()`` sampler.

    Parameters
    ----------
    low: Union[float, np.array of floats], default 0
        Lower bound of the output interval. All values generated will be greater than
        or equal to ``low``.
    high: Union[float, np.array of floats], default 1
        Upper bound of the output interval. All values generated will be less than or
        equal to ``high``.
    """

    def __init__(
        self, low: Union[float, np.array] = 0.0, high: Union[float, np.array] = 1.0
    ):
        super().__init__()
        self.low = low
        self.high = high

    def sample(self, size: tuple[int]) -> np.array:
        """Draw random samples from a uniform distribution.

        Parameters
        ----------
        size: tuple[int]
            Required shape of the random sample.
        """
        if len(size) != 2:
            ValueError(
                f"Expected number of input dimensions is 2, but were given {len(size)}."
            )
        return np.random.uniform(self.low, self.high, size)

# Base class for causal mechanism generation
class PredictionModel(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, X: np.array) -> np.array:
        raise NotImplementedError
    
# * Linear mechanisms *
class LinearMechanism(PredictionModel):
    """Linear causal mechanism by linear regression.

    Parameters
    ----------
    min_weight: float, default -1
        Minimum value for the coefficients of the linear mechanisms.
    max_weight: float, default 1
        Maximum value for the coefficients of the linear mechanisms.
    min_abs_weight: float, default 0.05
        Smallest allowed absolute value of any linear mechanism coefficient.
        Low value of ``min_abs_weight`` potentially lead to lambda-unfaithful distributions.
    """

    def __init__(
        self,
        min_weight: float = -1.0,
        max_weight: float = 1.0,
        min_abs_weight: float = 0.05,
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_abs_weight = min_abs_weight

        # One of max_weight or min_weight must be larger (abs value) than min_abs_weight
        if not (abs(max_weight) > min_abs_weight or abs(min_weight) > min_abs_weight):
            raise ValueError(
                "The range of admitted weights is empty. Please set"
                " one between ``min_weight`` and ``max_weight`` with absolute"
                "  value larger than ``min_abs_weight``."
            )

        self.linear_reg = LinearRegression(fit_intercept=False)

    def predict(self, X: np.array) -> np.array:
        """Apply a linera transformation on X.

        Given a vector ``x`` with :math:`p` features, the output ``y`` is given by:

        .. math::

                y = \sum_{i=1}^p \\alpha_i x_i

        where :math:`\\alpha_i` are random coefficients.

        Parameters
        ----------
        X: np.array, shape (num_samples, num_parents)
            Parents' observtations to be transformed by the causal mechanism.

        Returns
        -------
        y:  np.array, shape (num_samples)
            The output of the causal mechanism.
        """
        if X.ndim != 2:
            X = X.reshape((-1, 1))
        n_covariates = X.shape[1]

        # Random initialization of the causal mechanism
        self.linear_reg.coef_ = np.random.uniform(
            self.min_weight, self.max_weight, n_covariates
        )

        # Reject ~0 coefficients
        for i in range(n_covariates):
            while abs(self.linear_reg.coef_[i]) < self.min_abs_weight:
                self.linear_reg.coef_[i] = np.random.uniform(
                    self.min_weight, self.max_weight, 1
                )

        self.linear_reg.intercept_ = 0
        effect = self.linear_reg.predict(X)
        return effect


# * Nonlinear mechanisms *
class GaussianProcessMechanism(PredictionModel):
    """Nonlinear causal mechanism sampled from a Gaussian process.

    The nonlinear transformation is generated sampling the effect from a
    Gaussian process with covariance matrix defined as the kernel matrix of the
    parents' observations.

    Parameters
    ----------
    gamma: float, default 1
        The gamma parameters fixing the variance of the kernel.
        Larger values of gamma determines bigger magnitude of the causal mechanisms.
    """

    def __init__(self, gamma: float = 1.0):
        self.rbf = PairwiseKernel(gamma=gamma, metric="rbf")

    def predict(self, X: np.array) -> np.array:
        """Generate the effect given the observations of the parent nodes.

        The effect is generated as a nonlinear function sampled from a
        gaussian process.

        Parameters
        ----------
        X: np.array, shape (num_samples, num_parents)
            Input of the RBF kernel.

        Returns
        -------
        y: np.array, shape (num_samples)
            Causal effect sampled from the gaussian process with
            covariance matrix given by the RBF kernel with X as input.
        """
        num_samples = X.shape[0]
        # rbf = GPy.kern.RBF(input_dim=X.shape[1],lengthscale=self.lengthscale,variance=self.f_magn)
        # covariance_matrix = rbf.K(X,X)
        covariance_matrix = self.rbf(X, X)

        # Sample the effect as a zero centered normal with covariance given by the RBF kernel
        effect = np.random.multivariate_normal(np.zeros(num_samples), covariance_matrix)
        return effect

def topological_order(adjacency: np.array):
    # DAG test
    if not nx.is_directed_acyclic_graph(
        nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    ):
        raise ValueError("The input adjacency matrix is not acyclic.")

    # Define toporder one leaf at the time
    order = list()
    num_nodes = len(adjacency)
    mask = np.zeros((num_nodes))
    for _ in range(num_nodes):
        children_per_node = (
            adjacency.sum(axis=1) + mask - adjacency[:,order].sum(axis=1)
        )  # adjacency[i, j] = 1 --> i parent of j
        leaf = np.argmin(children_per_node)  # find leaf as node with no children
        mask[leaf] += float("inf")  # select node only once
        order.append(leaf)  # update order

    order = order[::-1]  # source first
    return order