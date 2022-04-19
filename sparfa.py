""" Sparse Factor Analysis was developed by Lan, Waters, Studer,
and Baraniuk (2014) to jointly infer student ability, item difficulty,
and item-to-concept relation (Q matrix) from response data.
This is an implementation of their approach in the sklearn/scipy
environment.

The original paper is here: https://www.jmlr.org/beta/papers/v15/lan14a.html

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
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

P_CLIP_     = 1E-8
P_CLIP_HI_  = 1. - P_CLIP_
LOGIT_CLIP_ = np.log(1. / P_CLIP_ - 1.)

class SPARFA:
    """ A SPARFA model describes student responses via item response
    theory with a Q matrix.

    In particular, we model the probability of student i answering
    correctly on item j via

    p[i, j] = sigma(np.dot(theta[i, :], Q[j, :]) - b[j])

    where theta[i, :] is the ability of student i on all
    domain-relevant skills, Q[j, :] describes which skills are
    relevant for item j, and b[j] is the difficulty of task j.
    Sigma denotes the logistic function.

    The model is learned via an alternating optimization.
    We first fit Q[j, :] and b[j] for each item j and then
    theta[i, :] for each student i. Each optimization is small
    and, thus, fairly fast.

    Parameters
    ----------
    num_concepts: int
        The number of skills or concepts in the domain.
    num_iterations: int (default = 10)
        The number of alternating optimization iterations.
    l1regul: float (default = 0.1)
        The L1 regularization strength to encourage sparsity in the
        Q matrix.
    l2regul: float (default = 1E-3)
        The L2 regularization strength.

    Attributes
    ----------
    Q_: ndarray
        A coupling matrix between items and concepts/skills, where
        Q_[j, k] indicates how relevant concept k is for item j.
        Q_ is encouraged to be sparse and forced to be non-negative.
    b_: ndarray
        The difficulty for each item.
    Theta_: ndarray
        An ability matrix where Theta_[i, k] indicates the ability
        of student i on skill/concept k.

    """
    def __init__(self, num_concepts, num_iterations = 10, l1regul = 0.1, l2regul = 1E-3):
        self.num_concepts = num_concepts
        self.num_iterations = num_iterations
        self.l1regul = l1regul
        self.l2regul = l2regul


    def fit(self, X, Y = None, verbose = False, ignore_optimizer_failures = True):
        """ Fits this model to the given response matrix.

        Parameters
        ----------
        X: ndarray
            A matrix where X[i, j] = 1 if student i answered item j
            correctly and X[i, j] = 0, otherwise.

        """
        m = X.shape[0]
        n = X.shape[1]
        # initialize Q based on a clustering of responses
        # impute nans first for that clustering
        Ximp = SimpleImputer(missing_values = np.nan, strategy = 'mean').fit_transform(X)
        clust = KMeans(n_clusters = self.num_concepts)
        clust.fit(Ximp.T)
        Q = np.zeros((n, self.num_concepts))
        for k in range(self.num_concepts):
            Q[clust.labels_ == k, k] = 1.
        # initialize b as zeros
        b = np.zeros(n)
        # initialize student knowledge by counting the number
        # of correct responses in each concept and z-normalizing
        # the resulting count
        Theta = np.dot(Ximp, Q)
        Theta = StandardScaler().fit_transform(Theta)
        # prepare aux variables
        pos   = X > 0.5
        neg   = X < 0.5
        present = np.logical_not(np.isnan(X))
        # now we can start the actual optimization loop
        for it in range(self.num_iterations):
            # optimize Q and b jointly via non-negative, l1 and l2
            # regularized logistic regression. Note that we perform
            # a separate optimization for each item
            loss = 0.
            for j in range(n):
                # prepare the objective function for optimization
                # of Q[j, :] and b[j]
                posj  = pos[:, j]
                negj  = neg[:, j]
                presj = present[:, j]
                xj    = X[presj, j]
                def objective(params):
                    qj = params[:-1]
                    bj = params[-1]
                    # compute logits
                    zj = np.dot(Theta, qj) - bj
                    # compute probabilities
                    pj = np.zeros_like(zj)
                    pj[zj >  LOGIT_CLIP_] = P_CLIP_HI_
                    pj[zj < -LOGIT_CLIP_] = P_CLIP_
                    non_clipped = np.logical_and(zj <= LOGIT_CLIP_, zj >= -LOGIT_CLIP_)
                    pj[non_clipped] = 1. / (1. + np.exp(-zj[non_clipped]))
                    # compute loss
                    l = -np.sum(np.log(pj[posj])) -np.sum(np.log(1. - pj[negj])) + self.l2regul * np.sum(np.square(qj)) + self.l1regul * np.sum(qj)
                    # compute gradient
                    delta = pj[presj] - xj
                    grad = np.zeros_like(params)
                    grad[:-1] = np.dot(delta, Theta[presj, :]) + 2 * self.l2regul * qj + self.l1regul * np.ones_like(qj)
                    grad[-1] = -np.sum(delta)
                    return l, grad
                # set up bounds
                bounds = [(0., np.inf)] * (self.num_concepts) + [(-np.inf, np.inf)]
                # optimize
                res = minimize(objective, np.zeros(self.num_concepts + 1) , jac = True, bounds = bounds)
                loss += res.fun
                if not ignore_optimizer_failures and not res.success:
                    raise ValueError('optimization for item %d failed with message %s' % (j, res.message))
                # store results and normalize every column (concept)
                # by its maximum value
                Q[j, :] = res.x[:-1]
                b[j]    = res.x[-1]
            if verbose:
                print('item loss after iteration %d: %g' % (it + 1, loss))

            # opimize Theta based on current Q and b. We do that
            # independently for each student
            loss_ability = 0.
            for i in range(m):
                # prepare the objective function for optimization
                # of Theta[i, :]
                posi  = pos[i, :]
                negi  = neg[i, :]
                presi = present[i, :]
                xi    = X[i, presi]
                Qi    = Q[presi, :]
                def objective(thetai):
                    # compute logits
                    zi = np.dot(thetai, Q.T) - b
                    # compute probabilities
                    pi = np.zeros_like(zi)
                    pi[zi > LOGIT_CLIP_]  = P_CLIP_HI_
                    pi[zi < -LOGIT_CLIP_] = P_CLIP_
                    non_clipped = np.logical_and(zi <= LOGIT_CLIP_, zi >= -LOGIT_CLIP_)
                    pi[non_clipped] = 1. / (1. + np.exp(-zi[non_clipped]))
                    # compute loss
                    l = -np.sum(np.log(pi[posi])) - np.sum(np.log(1. - pi[negi])) + self.l2regul * np.sum(np.square(thetai))
                    # compute gradient
                    delta = pi[presi] - xi
                    grad  = np.dot(delta, Qi) + 2 * self.l2regul * thetai
                    return l, grad
                # optimize
                res = minimize(objective, Theta[i, :], jac = True)
                loss_ability += res.fun
                if not ignore_optimizer_failures and not res.success:
                    raise ValueError('optimization for student %d failed with message %s' % (i, res.message))
                Theta[i, :] = res.x
            if verbose:
                print('ability loss after iteration %d: %g' % (it + 1, loss_ability))

#            # optimize Theta based on current Q and b. We do that in
#            # one big chunk for the sake of computational efficiency.
#            def objective(params):
#                Theta = params.reshape((m, self.num_concepts))
#                # compute logits
#                Z = np.dot(Theta, Q.T) - np.expand_dims(b, 0)
#                # compute probabilities
#                P = np.zeros_like(Z)
#                P[Z > LOGIT_CLIP_]  = P_CLIP_HI_
#                P[Z < -LOGIT_CLIP_] = P_CLIP_
#                non_clipped = np.logical_and(Z <= LOGIT_CLIP_, Z >= -LOGIT_CLIP_)
#                P[non_clipped] = 1. / (1. + np.exp(-Z[non_clipped]))
#                # compute loss
#                l = -np.sum(np.log(P[pos])) - np.sum(np.log(1. - P[neg])) + self.l2regul * np.sum(np.square(Theta))
#                # compute gradient
#                Delta = P - X
#                Delta[np.isnan(X)] = 0.
#                grad = np.dot(Delta, Q) + 2 * self.l2regul * Theta
#                return l, np.ravel(grad)
#            # start optimization
#            res = minimize(objective, np.ravel(Theta), jac = True)
#            print('ability loss after iteration %d: %g' % (it + 1, res.fun))
#            if not res.success:
#                raise ValueError('optimization for abilities failed with message %s' % res.message)
#            Theta = res.x.reshape((m, self.num_concepts))
        # store final results
        self.Q_ = Q
        self.b_ = b
        self.Theta_ = Theta

    def encode(self, X, ignore_optimizer_failures = True):
        """ Infers the knowledge of each student based on their
        given response patterns. Note that this function performs
        an optimization.

        Parameters
        ----------
        X: ndarray
            A response matrix where X[i, j] = 1 if student i answered
            item j correctly and X[i, j] = 0, otherwise.

        Returns
        -------
        Theta: ndarray
            An ability matrix where Theta[i, k] indicates the ability
            of student i on skill/concept k.

        """
        m, n = X.shape
        Theta = np.zeros((m, self.num_concepts))
        for i in range(m):
            # prepare the objective function for optimization
            # of Theta[i, :]
            posi  = X[i, :] > 0.5
            negi  = X[i, :] < 0.5
            presi = np.logical_not(np.isnan(X[i, :]))
            xi    = X[i, presi]
            Qi    = self.Q_[presi, :]
            def objective(thetai):
                # compute logits
                zi = np.dot(thetai, self.Q_.T) - self.b_
                # compute probabilities
                pi = np.zeros_like(zi)
                pi[zi > LOGIT_CLIP_]  = P_CLIP_HI_
                pi[zi < -LOGIT_CLIP_] = P_CLIP_
                non_clipped = np.logical_and(zi <= LOGIT_CLIP_, zi >= -LOGIT_CLIP_)
                pi[non_clipped] = 1. / (1. + np.exp(-zi[non_clipped]))
                # compute loss
                l = -np.sum(np.log(pi[posi])) - np.sum(np.log(1. - pi[negi])) + self.l2regul * np.sum(np.square(thetai))
                # compute gradient
                delta = pi[presi] - xi
                grad  = np.dot(delta, Qi) + 2 * self.l2regul * thetai
                return l, grad
            # optimize
            res = minimize(objective, Theta[i, :], jac = True)
            if not ignore_optimizer_failures and not res.success:
                raise ValueError('optimization for student %d failed with message %s' % (i, res.message))
            Theta[i, :] = res.x
        return Theta


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

    def predict(self, X = None):
        """ Predicts whether students will answer correctly on each
        item according to the trained model.

        Parameters
        ----------
        X: ndarray (default = None)
            A response matrix where X[i, j] = 1 if student i answered
            item j correctly and X[i, j] = 0, otherwise.
            If not given, the prediction is made on the training
            data.

        Returns
        -------
        X: ndarray
            A response matrix where X[i, j] = 1 if student i is
            preidcted to answer item j correctly and X[i, j] = 0,
            otherwise.

        """
        # compute knowledge
        if X is None:
            Theta = self.Theta_
        else:
            Theta = self.encode(X)
        # decode
        return self.decode(Theta)

    def predict_proba(self, X = None):
        """ Predicts whether students will answer correctly on each
        item according to the trained model.

        Parameters
        ----------
        X: ndarray (default = None)
            A response matrix where X[i, j] = 1 if student i answered
            item j correctly and X[i, j] = 0, otherwise.
            If not given, the prediction is made on the training
            data.

        Returns
        -------
        P: ndarray
            A matrix where P[i, j] indicates the predicted probability
            of student i to answer item j correctly.

        """
        # compute knowledge
        if X is None:
            Theta = self.Theta_
        else:
            Theta = self.encode(X)
        # decode
        return self.decode_proba(Theta)

    def Q(self):
        return self.Q_

    def difficulties(self):
        return self.b_

## sample ground-truth knowledge, item difficulties, and Q matrix
#m = 100
#n = 20
#K = 5
#Theta = np.random.randn(m, K)
#Q     = np.zeros((n, K))
#for j in range(n):
#    Q[j, np.random.choice(K, size = 2, replace = False)] = np.random.rand(2)
#b     = np.random.randn(n)

#Ztrue = np.dot(Theta, Q.T) - b
#Ptrue = 1. / (1. + np.exp(-Ztrue))
#Xtrue = np.random.rand(m, n)
#Xtrue[Xtrue >= 1. - Ptrue] = 1.
#Xtrue[Xtrue <  1. - Ptrue] = 0.

## include some nans
#for i in range(m):
#    Xtrue[i, np.random.choice(n, size = 1)] = np.nan

## apply model
#model = SPARFA(K, num_iterations = 5, l2regul = 1.)
#model.fit(Xtrue)

## check accuracy
#X = model.predict()
#print('model error: %g' % np.nanmean(np.abs(X - Xtrue)))

## use hungarian algorithm to match learned concepts and
## ground truth concepts
#import matplotlib.pyplot as plt
#from scipy.optimize import linear_sum_assignment
#from scipy.spatial.distance import cdist

#Qnorm = model.Q_ / np.expand_dims(np.max(model.Q_, 0), 0)

#Costs = cdist(Q.T, Qnorm.T) ** 2
#rows, cols = linear_sum_assignment(Costs)

#plt.imshow(Costs[:, cols])
#plt.colorbar()
#plt.title('Q matrix distance (err: %g)' %  np.sum(Costs[rows, cols]))
#plt.show()

## show relation between predicted item difficulty and success
## rate as well as predicted ability and success rate
#plt.figure(figsize=(16, 10))
#plt.subplot(2, 2, 1)
#plt.scatter(model.b_, b)
#plt.xlabel('predicted difficulty')
#plt.ylabel('actual difficulty')
#plt.subplot(2, 2, 2)
#plt.scatter(model.b_, np.nanmean(Xtrue, 0))
#plt.xlabel('predicted difficulty')
#plt.ylabel('success rate')
#plt.subplot(2, 2, 3)
#plt.scatter(np.mean(model.Theta_, 1), np.mean(Theta, 1))
#plt.xlabel('predicted mean ability')
#plt.ylabel('actual mean ability')
#plt.title('student ability')
#plt.subplot(2, 2, 4)
#plt.scatter(np.mean(model.Theta_, 1), np.nanmean(Xtrue, 1))
#plt.xlabel('predicted mean ability')
#plt.ylabel('success rate')
#plt.title('student ability')
#plt.show()

