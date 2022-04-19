"""
our model is that a student's knowledge theta
can be inferred from a student's responses by
multiplying with the Q matrix and, conversely, the
Q matrix also maps knowledge to response logits.
In other words, our model is captured in the following
two equations:
theta_i = Q.T * x_i
z_i = Q * theta_i - b
where theta_i is the knowledge of student i, x_i are
the responses of student i in the test, b are the task
difficulties, and z_i are the response logits of student i.

Our model parameters are Q and b. We regularize Q with both
L2 and L1 penalites to ensure sparsity and we also ensure
non-negativity of Q, further encouraging sparsity.
We penalize b with l2 regularization.

"""

# Sparse Factor Autoencoders for Item Response Theory
# Copyright (C) 2021-2022
# Benjamin Paaßen
# German Research Center for Artificial Intelligence
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2021-2022, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '0.1.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@dfki.de'

import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import minimize
from sklearn.cluster import KMeans

P_CLIP_     = 1E-8
P_CLIP_HI_  = 1. - P_CLIP_
LOGIT_CLIP_ = np.log(1. / P_CLIP_ - 1.)

class QFactorModel:
    """
    A log-linear autoencoder model for item response theory.

    Student knowledge is modeled as
    theta = Q.T * x
    where Q is the coupling matrix between items and concepts
    and x is the vector of student responses.

    Responses are modeled as
    p = Q * theta - b
    where b is the vector of item difficulties.

    In other words, this model tries to 'autoencode' the student's
    responses via the mapping f(x) = sigma(Q * Q.T * x - b),
    where sigma is the logistic map.

    The parameters of the model are the Q matrix and the difficulty
    vector b. 
    
    """
    def __init__(self, num_concepts, l1regul = 1., l2regul = 1.):
        self.num_concepts = num_concepts
        self.l1regul = l1regul
        self.l2regul = l2regul

    def fit(self, X):
        m, n = X.shape
        K = self.num_concepts

        # impute nans
        self.p_ = np.nanmean(X, 0)
        nans = np.isnan(X)
        if np.any(nans):
            X = np.copy(X)
            for j in range(n):
                if np.any(nans[:, j]):
                    X[nans[:, j], j] = np.random.binomial(1, size = int(np.sum(nans[:, j])), p = self.p_[j])


        # initialize Q based on a clustering of responses
        # impute nans first for that clustering
        clust = KMeans(n_clusters = K)
        clust.fit(X.T)
        Q = np.zeros((n, K))
        for k in range(K):
            Q[clust.labels_ == k, k] = 1.


        # set up aux variables
        pos = X > 0.5
        neg = X < 0.5
        l1grad = np.ones_like(Q)

        # set up objective function
        def objective(params):
            # extract Q and b
            Q = np.reshape(params[:n*K], (n, K))
            b = params[n*K:]
            # compute student knowledge
            Theta = np.dot(X, Q)
            # compute logits
            Z = np.dot(Theta, Q.T) - np.expand_dims(b, 0)
            # compute probabilities
            P = np.zeros_like(Z)
            P[Z >  LOGIT_CLIP_] = P_CLIP_HI_
            P[Z < -LOGIT_CLIP_] = P_CLIP_
            non_clipped = np.logical_and(Z <= LOGIT_CLIP_, Z >= -LOGIT_CLIP_)
            P[non_clipped] = 1. / (1. + np.exp(-Z[non_clipped]))
            # compute loss
            l = -np.sum(np.log(P[pos])) -np.sum(np.log(1. - P[neg])) + self.l2regul * np.sum(np.square(Q)) + self.l1regul * np.sum(Q) + self.l2regul * np.sum(np.square(b))
            # compute gradients
            Delta = P - X
            gradQ = np.einsum('ij,jk,ir->rk', Delta, Q, X) + np.dot(Delta.T, Theta) + 2 * self.l2regul * Q + self.l1regul * l1grad
            gradb = -np.sum(Delta, 0) + 2 * self.l2regul * b
            # return
            grad = np.concatenate((np.ravel(gradQ), gradb))
            return l, grad

        # set up bounds
        bounds = [(0., np.inf)] * ((K+1) * n)
        # optimize
        params_init = np.concatenate((np.ravel(Q), np.zeros(n)))
        res = minimize(objective, params_init, jac = True, bounds = bounds)
        if not res.success:
            print(res)
            raise ValueError('optimization failed with message %s' % res.message)
        # store the results
        self.Q_ = np.reshape(res.x[:n*K], (n, K))
        self.b_ = res.x[n*K:]

        return self

    def encode(self, X):
        """ Encodes the given student response matrix to knowledge.

        Parameters
        ----------
        X: ndarray
            A m x n matrix where X[i, j] = 1 if student i responds
            correctly to item j and X[i, j] = 0, otherwise.

        Returns
        -------
        Theta: ndarray
            A m x K matrix, where Theta[i, k] corresponds to the
            estimated knowledge of student i on concept k.

        """
        nans = np.isnan(X)
        if np.any(nans):
            X = np.copy(X)
            for j in range(X.shape[1]):
                if np.any(nans[:, j]):
                    X[nans[:, j], j] = self.p_[j]
        return np.dot(X, self.Q_)


    def decode(self, Theta):
        """ Predicts correct responses for each student on
        each item based on the given knowledge matrix.

        Parameters
        ----------
        Theta: ndarray
            An ability matrix where Theta[i, k] indicates the ability
            of student i on skill/concept k.

        Returns
        -------
        X: ndarray
            A response matrix where X[i, j] indicates the predicted
            logit of the probability that student i answers correctly
            on item j.

        """
        # compute logits
        Z = np.dot(Theta, self.Q_.T) - np.expand_dims(self.b_, 0)
        # binarize
        Z[Z <= 0.] = 0.
        Z[Z >  0.] = 1.
        return Z


    def decode_proba(self, Theta):
        """ Predicts repsonse probabilities for each student on
        each item based on the given knowledge matrix.

        Parameters
        ----------
        Theta: ndarray
            An ability matrix where Theta[i, k] indicates the ability
            of student i on skill/concept k.

        Returns
        -------
        P: ndarray
            A matrix where P[i, j] indicates the predicted probability
            of student i to answer item j correctly.

        """
        # compute logits
        Z = np.dot(Theta, self.Q_.T) - np.expand_dims(self.b_, 0)
        # compute probabilities
        P = np.zeros_like(Z)
        P[Z > LOGIT_CLIP_]  = P_CLIP_HI_
        P[Z < -LOGIT_CLIP_] = P_CLIP_
        non_clipped = np.logical_and(Z <= LOGIT_CLIP_, Z >= -LOGIT_CLIP_)
        P[non_clipped] = 1. / (1. + np.exp(-Z[non_clipped]))
        return P


    def predict(self, X):
        """ Predicts whether students will answer correctly on each
        item according to the trained model.

        Parameters
        ----------
        X: ndarray
            A response matrix where X[i, j] = 1 if student i answered
            item j correctly and X[i, j] = 0, otherwise.

        Returns
        -------
        X: ndarray
            A response matrix where X[i, j] = 1 if student i is
            preidcted to answer item j correctly and X[i, j] = 0,
            otherwise.

        """
        # compute knowledge
        Theta = self.encode(X)
        # decode
        return self.decode(Theta)


    def predict_proba(self, X):
        """ Predicts whether students will answer correctly on each
        item according to the trained model.

        Parameters
        ----------
        X: ndarray
            A response matrix where X[i, j] = 1 if student i answered
            item j correctly and X[i, j] = 0, otherwise.

        Returns
        -------
        P: ndarray
            A matrix where P[i, j] indicates the predicted probability
            of student i to answer item j correctly.

        """
        # compute knowledge
        Theta = self.encode(X)
        # decode
        return self.decode_proba(Theta)


    def Q(self):
        return self.Q_

    def difficulties(self):
        return self.b_



class QTwoFactorModel:
    """
    A log-linear autoencoder model for item response theory.

    Student knowledge is modeled as
    theta = A * x
    where A is the coupling matrix between items and concepts
    and x is the vector of student responses.

    Responses are modeled as
    p = Q * theta - b
    where Q is another coupling matrix between concepts and items,
    and where b is the vector of item difficulties.

    In other words, this model tries to 'autoencode' the student's
    responses via the mapping f(x) = sigma(Q * A * x - b),
    where sigma is the logistic map.

    The parameters of the model are the matrices A and Q and the
    difficulty vector b. 

    """
    def __init__(self, num_concepts, l1regul = 1., l2regul = 1.):
        self.num_concepts = num_concepts
        self.l1regul = l1regul
        self.l2regul = l2regul

    def fit(self, X, ignore_optimization_errors = True):
        m, n = X.shape
        K = self.num_concepts

        # impute nans
        self.p_ = np.nanmean(X, 0)
        nans = np.isnan(X)
        if np.any(nans):
            X = np.copy(X)
            for j in range(n):
                if np.any(nans[:, j]):
                    X[nans[:, j], j] = np.random.binomial(1, size = int(np.sum(nans[:, j])), p = self.p_[j])


        # initialize A and Q based on a clustering of responses
        # impute nans first for that clustering
        clust = KMeans(n_clusters = K)
        clust.fit(X.T)
        Q = np.zeros((n, K))
        for k in range(K):
            Q[clust.labels_ == k, k] = 1.
        A = np.copy(Q.T)

        # set up aux variables
        pos = X > 0.5
        neg = X < 0.5
        l1grad = np.ones_like(Q)

        # set up objective function
        def objective(params):
            # extract A, Q, and b
            A = np.reshape(params[:n*K], (K, n))
            Q = np.reshape(params[n*K:2*n*K], (n, K))
            b = params[2*n*K:]
            # compute student knowledge
            Theta = np.dot(X, A.T)
            # compute logits
            Z = np.dot(Theta, Q.T) - np.expand_dims(b, 0)
            # compute probabilities
            P = np.zeros_like(Z)
            P[Z >  LOGIT_CLIP_] = P_CLIP_HI_
            P[Z < -LOGIT_CLIP_] = P_CLIP_
            non_clipped = np.logical_and(Z <= LOGIT_CLIP_, Z >= -LOGIT_CLIP_)
            P[non_clipped] = 1. / (1. + np.exp(-Z[non_clipped]))
            # compute loss
            l = -np.sum(np.log(P[pos])) -np.sum(np.log(1. - P[neg])) + self.l2regul * np.sum(np.square(A)) + self.l1regul * np.sum(A) + + self.l2regul * np.sum(np.square(Q)) + self.l1regul * np.sum(Q) + self.l2regul * np.sum(np.square(b))
            # compute gradients
            Delta = P - X
            gradA = np.einsum('ij,jk,ir->rk', Delta, Q, X).T + 2 * self.l2regul * A + self.l1regul * l1grad.T
            gradQ = np.dot(Delta.T, Theta) + 2 * self.l2regul * Q + self.l1regul * l1grad
            gradb = -np.sum(Delta, 0) + 2 * self.l2regul * b
            # return
            grad = np.concatenate((np.ravel(gradA), np.ravel(gradQ), gradb))
            return l, grad

        # set up bounds
        bounds = [(0., np.inf)] * ((2*K+1) * n)
        # optimize
        params_init = np.concatenate((np.ravel(A), np.ravel(Q), np.zeros(n)))
        res = minimize(objective, params_init, jac = True, bounds = bounds)
        if not ignore_optimization_errors and not res.success:
            raise ValueError('optimization failed with message %s' % res.message)
        # store the results
        self.A_ = np.reshape(res.x[:n*K], (K, n))
        self.Q_ = np.reshape(res.x[n*K:2*n*K], (n, K))
        self.b_ = res.x[2*n*K:]

        return self

    def encode(self, X):
        """ Encodes the given student response matrix to knowledge.

        Parameters
        ----------
        X: ndarray
            A m x n matrix where X[i, j] = 1 if student i responds
            correctly to item j and X[i, j] = 0, otherwise.

        Returns
        -------
        Theta: ndarray
            A m x K matrix, where Theta[i, k] corresponds to the
            estimated knowledge of student i on concept k.

        """
        nans = np.isnan(X)
        if np.any(nans):
            X = np.copy(X)
            for j in range(X.shape[1]):
                if np.any(nans[:, j]):
                    X[nans[:, j], j] = self.p_[j]
        return np.dot(X, self.A_.T)


    def decode(self, Theta):
        """ Predicts correct responses for each student on
        each item based on the given knowledge matrix.

        Parameters
        ----------
        Theta: ndarray
            An ability matrix where Theta[i, k] indicates the ability
            of student i on skill/concept k.

        Returns
        -------
        X: ndarray
            A response matrix where X[i, j] indicates the predicted
            logit of the probability that student i answers correctly
            on item j.

        """
        # compute logits
        Z = np.dot(Theta, self.Q_.T) - np.expand_dims(self.b_, 0)
        # binarize
        Z[Z <= 0.] = 0.
        Z[Z >  0.] = 1.
        return Z


    def decode_proba(self, Theta):
        """ Predicts repsonse probabilities for each student on
        each item based on the given knowledge matrix.

        Parameters
        ----------
        Theta: ndarray
            An ability matrix where Theta[i, k] indicates the ability
            of student i on skill/concept k.

        Returns
        -------
        P: ndarray
            A matrix where P[i, j] indicates the predicted probability
            of student i to answer item j correctly.

        """
        # compute logits
        Z = np.dot(Theta, self.Q_.T) - np.expand_dims(self.b_, 0)
        # compute probabilities
        P = np.zeros_like(Z)
        P[Z > LOGIT_CLIP_]  = P_CLIP_HI_
        P[Z < -LOGIT_CLIP_] = P_CLIP_
        non_clipped = np.logical_and(Z <= LOGIT_CLIP_, Z >= -LOGIT_CLIP_)
        P[non_clipped] = 1. / (1. + np.exp(-Z[non_clipped]))
        return P


    def predict(self, X):
        """ Predicts whether students will answer correctly on each
        item according to the trained model.

        Parameters
        ----------
        X: ndarray
            A response matrix where X[i, j] = 1 if student i answered
            item j correctly and X[i, j] = 0, otherwise.

        Returns
        -------
        X: ndarray
            A response matrix where X[i, j] = 1 if student i is
            preidcted to answer item j correctly and X[i, j] = 0,
            otherwise.

        """
        # compute knowledge
        Theta = self.encode(X)
        # decode
        return self.decode(Theta)


    def predict_proba(self, X):
        """ Predicts whether students will answer correctly on each
        item according to the trained model.

        Parameters
        ----------
        X: ndarray
            A response matrix where X[i, j] = 1 if student i answered
            item j correctly and X[i, j] = 0, otherwise.

        Returns
        -------
        P: ndarray
            A matrix where P[i, j] indicates the predicted probability
            of student i to answer item j correctly.

        """
        # compute knowledge
        Theta = self.encode(X)
        # decode
        return self.decode_proba(Theta)

    def Q(self):
        return self.Q_

    def difficulties(self):
        return self.b_
