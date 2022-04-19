""" An implementation of variational inference as described by
Wu et al. (2020):
https://educationaldatamining.org/files/conferences/EDM2020/papers/paper_22.pdf
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
import torch


class VIBO_Q(torch.nn.Module, BaseEstimator):
    """ A variational inference scheme for item response theory.
    Instead of directly inferring latent parameters via maximum
    likelihood, we train a surrogate distribution q for the posterior
    of the latent variables and train q to maximize the marginal
    probability of observable variables; or, more precisely, the
    variational lower bound (VIBO) of this marginal probability.

    In more detail, we first sample item difficulties from a
    Gaussian distribution with learned mean and variance. Then,
    we sample student abilities from a conditional Gaussian
    whose mean is computed via a linear layer from the sampled
    item difficulties and the observed variables.
    Finally, we compute the log probability
    of the observed responses given the sampled difficulties and
    abilities and adjust all distribution parameters to maximize
    the VIBO, which is equivalent to the log probability plus
    the kullback leibler divergences between the difficulty/ability
    distributions and a standard normal distribution.

    Parameters
    ----------
    Q: ndarray
        A Q matrix mapping items to skills.
    lr: float (default = 5E-3)
        The learning rate for fitting the data.
    num_epochs: int (default = 100)
        The number of training epochs. Each epoch is a run over
        the entire data set
    minibatch_size: int (default = 16)
        The size of each training brach.
    regul: float (default = 1.)
        The weight for the Kullback-Leibler divergence in the
        variational bound.

    Attributes
    ----------
    q_b: class torch.nn.Embedding
        An embedding layer to store the means and log-variances
        for the item difficulty distributions.
    q_theta: class torch.nn.Linear
        A linear layer to map difficulties and item responses to
        the mean and log-variance of the ability distribution
        for each skill.

    """
    def __init__(self, Q, lr = 5E-3, num_epochs = 100, minibatch_size = 16, regul = 1.):
        super(VIBO_Q, self).__init__()
        self.Q_ = Q
        self.Qtorch = torch.tensor(Q, dtype=torch.float)
        self.lr = lr
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.regul = regul
        # initialize the torch layers
        self.q_b = torch.nn.Embedding(Q.shape[0], 2)
        self.q_theta = torch.nn.Linear(Q.shape[0] * 2, 2 * Q.shape[1])


    def forward(self, X):
        """ A synonym for 'encode'.

        Parameters
        ----------
        X: ndarray
            A matrix of test responses where each row represents
            a student and each column represents a question.

        Returns
        -------
        theta: ndarray
            A matrix of predicted abilities where each row represents
            a student an each column represents a skill.

        """
        return self.encode(X)

    def encode(self, X):
        """ Predicts student ability from student responses.

        Parameters
        ----------
        X: ndarray
            A matrix of test responses where each row represents
            a student and each column represents a question.

        Returns
        -------
        theta: ndarray
            A matrix of predicted abilities where each row represents
            a student an each column represents a skill.

        """
        # check that the dimensions fit with the input data
        if X.shape[1] != self.Q_.shape[0]:
            raise ValueError('Expected one column in X for each row in Q.')
        X = torch.tensor(X, dtype=torch.float)
        # use the item distribution mean as difficulty
        b = self.q_b.weight[:, 0]
        # repeat for every student
        B = b.repeat(X.shape[0], 1)
        # mask out nans in input
        nanmask = torch.isnan(X)
        B[nanmask] = 0.
        X[nanmask] = 0.
        # concatenate X and B and acquire ability estimates;
        # again, we use only the mean for encoding
        Theta = self.q_theta(torch.cat((X, B), 1))[:, :self.Q_.shape[1]]
        return Theta.detach().numpy()

    def decode(self, Theta):
        """ Decodes the given knowledge into predicted
        test results.

        Parameters
        ----------
        Theta: ndarray
            A matrix with one row per student and one column per
            skill, where Theta[i, k] represents the estimated
            knowledge of student i for skill k.

        Returns
        -------
        Y: ndarray
            A matrix of predicted test responses for each student
            on each item.

        """
        # multiply Theta with Q to get the relevant knowledge for
        # each item
        Theta_hat = np.dot(Theta, self.Q_.T)
        # subtract the difficulties
        # use the item distribution mean as difficulty
        b = self.q_b.weight[:, 0]
        Y = Theta_hat - np.expand_dims(b.detach().numpy(), 0)
        # binarize result
        Y[Y <= 0.] = 0.
        Y[Y > 0.]  = 1.
        return Y


    def decode_proba(self, Theta):
        """ Decodes the given knowledge into success probabilities.

        Parameters
        ----------
        Theta: ndarray
            A matrix with one row per student and one column per
            skill, where Theta[i, k] represents the estimated
            knowledge of student i for skill k.

        Returns
        -------
        P: ndarray
            A matrix of predicted success probabilities for each
            student on each item.

        """
        # multiply Theta with Q.T to get the relevant knowledge for
        # each item
        Theta_hat = np.dot(Theta, self.Q_.T)
        # subtract the difficulties
        # use the item distribution mean as difficulty
        b = self.q_b.weight[:, 0]
        Y = Theta_hat - np.expand_dims(b.detach().numpy(), 0)
        # apply logistic function
        return 1. / (1. + np.exp(-Y))


    def predict(self, X):
        """ Auto-encodes the given test results.

        Parameters
        ----------
        X: ndarray
            A matrix of test responses where each row represents
            a student and each column represents a question.

        Returns
        -------
        Y: ndarray
            A matrix of predicted test responses for each student
            on each item.

        """
        Theta = self.encode(X)
        return self.decode(Theta)


    def predict_proba(self, X):
        """ Auto-encodes the given test results.

        Parameters
        ----------
        X: ndarray
            A matrix of test responses where each row represents
            a student and each column represents a question.

        Returns
        -------
        P: ndarray
            A matrix of predicted success probabilities for each
            student on each item.

        """
        Theta = self.encode(X)
        return self.decode_proba(Theta)


    def compute_loss(self, X):
        """ Computes the VIBO loss for the given responses.

        Parameters
        ----------
        X: class torch.Tensor
            A matrix of test responses where each row represents
            a student and each column represents a question.

        Returns
        -------
        loss: class torch.tensor
            The VIBO loss.

        """
        # sample item difficulties first
        mu_b     = self.q_b.weight[:, 0]
        logvar_b = self.q_b.weight[:, 1]
        std_b    = torch.exp(0.5 * logvar_b)
        b        = torch.randn_like(mu_b) * std_b + mu_b

        # repeat for every student
        B = b.repeat(X.shape[0], 1)
        # mask out nans in input
        X = X.clone().detach()
        nanmask = torch.isnan(X)
        B[nanmask] = 0.
        X[nanmask] = 0.

        # sample abilities next
        MuLogvar     = self.q_theta(torch.cat((X, B), 1))
        Mu_theta     = MuLogvar[:, :self.Q_.shape[1]]
        Logvar_theta = MuLogvar[:, self.Q_.shape[1]:]
        Std_theta    = torch.exp(0.5 * Logvar_theta)
        Theta        = torch.randn_like(Mu_theta) * Std_theta + Mu_theta

        # Compute logits for each response probability
        Logits       = torch.mm(Theta, self.Qtorch.T) - B
        # mask out nans
        Logits[nanmask] = -100.

        # compute VIBO. First, we compute the binary crossentropy loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Logits, X)

        # add regularization/KL divergences
        if self.regul > 0.:
            loss = loss + .5 * self.regul * torch.mean(torch.square(mu_b) + torch.square(std_b) - logvar_b - 1.) + .5 * self.regul * torch.mean(torch.square(Mu_theta) + torch.square(Std_theta) - Logvar_theta - 1.)
        return loss


    def fit(self, X, Y = None, print_step = 0):
        """ Fits a model to the given response data.

        Parameters
        ----------
        X: ndarray
            A matrix of test responses where each row represents
            a student and each column represents a question.
        Y: ndarray (default = None)
            Not needed. Only here for consistency with sklearn
            interface.

        Returns
        -------
        self

        """
        # check that each entry of the Q matrix is zero or 1
        if np.any(np.abs(np.abs(self.Q_) + np.abs(self.Q_ - 1) - 1) > 1E-3):
            raise ValueError('The Q Matrix needs to be binary.')
        # check that each row contains exactly one one
        if np.any(np.abs(1 - np.sum(self.Q_, 1)) > 1E-3):
            raise ValueError('Each row in the Q matrix needs to contain exactly a single one.')
        # check that the dimensions fit with the training data
        if X.shape[1] != self.Q_.shape[0]:
            raise ValueError('Expected one column in X for each row in Q.')

        X = torch.tensor(X, dtype=torch.float)

        # set up ADAM optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # start training
        for epoch in range(self.num_epochs):
            # set up a random permutation of the data
            perm = torch.randperm(X.shape[0])
            # iterate over all data as minibatches
            for k in range(0, X.shape[0], self.minibatch_size):
                optimizer.zero_grad()
                # sample a minibatch of data
                minibatch   = perm[k:k+self.minibatch_size]
                X_minibatch = X[minibatch, :]
                # compute the current loss for it
                loss = self.compute_loss(X_minibatch)
                # compute gradient
                loss.backward()
                # perform an optimizer step
                optimizer.step()
            # print current state if so requested
            if print_step > 0 and (epoch+1) % print_step == 0:
                print('loss after %d epochs: %g' % (epoch+1, loss.item()))

        return self

    def Q(self):
        return self.Q_

    def difficulties(self):
        return self.q_b.weight[:, 0].detach().numpy()




class VIBO(torch.nn.Module, BaseEstimator):
    """ A variational inference scheme for item response theory.
    Instead of directly inferring latent parameters via maximum
    likelihood, we train a surrogate distribution q for the posterior
    of the latent variables and train q to maximize the marginal
    probability of observable variables; or, more precisely, the
    variational lower bound (VIBO) of this marginal probability.

    In more detail, we first sample item difficulties from a
    Gaussian distribution with learned mean and variance. Then,
    we sample student abilities from a conditional Gaussian
    whose mean is computed via a linear layer from the sampled
    item difficulties and the observed variables.
    Finally, we compute the log probability
    of the observed responses given the sampled difficulties and
    abilities and adjust all distribution parameters to maximize
    the VIBO, which is equivalent to the log probability plus
    the kullback leibler divergences between the difficulty/ability
    distributions and a standard normal distribution.

    Parameters
    ----------
    num_items: int
        The number of items.
    num_concepts: int
        The number of concepts in this domain.
    num_hidden: int (default = 0)
        The number of hidden neurons for encoding from difficulties
        and item responses to student ability. If zero (or negative),
        no hidden layer is used.
    lr: float (default = 5E-3)
        The learning rate for fitting the data.
    num_epochs: int (default = 100)
        The number of training epochs. Each epoch is a run over
        the entire data set
    minibatch_size: int (default = 16)
        The size of each training brach.
    regul: float (default = 1.)
        The weight for the Kullback-Leibler divergence in the
        variational bound.

    Attributes
    ----------
    q_b: class torch.nn.Embedding
        An embedding layer to store the means and log-variances
        for the item difficulty distributions.
    q_hidden: class torch.nn.Linear
        A linear layer to map difficulties and item responses
        to a hidden neuron layer. Only if num_hidden > 0.
    q_theta: class torch.Module
        A neural net to map difficulties and item responses to
        the mean and log-variance of the ability distribution
        for each skill. If num_hidden <= 0, this is just a single
        linear layer. Otherwise, it is a Sequential object with
        two layers and an intermediate sigmoid.
    p_theta: class torch.nn.Linear
        A linear layer to map

    """
    def __init__(self, num_items, num_concepts, num_hidden = 0, lr = 5E-3, num_epochs = 100, minibatch_size = 16, regul = 1.):
        super(VIBO, self).__init__()
        self.num_items = num_items
        self.num_concepts = num_concepts
        self.num_hidden = num_hidden
        self.lr = lr
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.regul = regul
        # initialize the encoder layers
        self.q_b = torch.nn.Embedding(self.num_items, 2)
        if self.num_hidden > 0:
            self.q_theta = torch.nn.Sequential(
                torch.nn.Linear(self.num_items * 2, self.num_hidden),
                torch.nn.Sigmoid(),
                torch.nn.Linear(self.num_hidden, self.num_concepts * 2))
        else:
            self.q_theta = torch.nn.Linear(self.num_items * 2, self.num_concepts * 2)
        # initialize the decoder layer. We don't use a bias here
        # because the bias corresponds to the difficulty.
        self.p_theta = torch.nn.Linear(self.num_concepts, self.num_items, bias = False)
        

    def forward(self, X):
        """ A synonym for 'encode'.

        Parameters
        ----------
        X: ndarray
            A matrix of test responses where each row represents
            a student and each column represents a question.

        Returns
        -------
        theta: ndarray
            A matrix of predicted abilities where each row represents
            a student an each column represents a skill.

        """
        return self.encode(X)

    def encode(self, X):
        """ Predicts student ability from student responses.

        Parameters
        ----------
        X: ndarray
            A matrix of test responses where each row represents
            a student and each column represents a question.

        Returns
        -------
        theta: ndarray
            A matrix of predicted abilities where each row represents
            a student an each column represents a skill.

        """
        # check that the dimensions fit with the input data
        if X.shape[1] != self.num_items:
            raise ValueError('Expected one column in X for each item.')
        X = torch.tensor(X, dtype=torch.float)
        # use the item distribution mean as difficulty
        b = self.q_b.weight[:, 0]
        # repeat for every student
        B = b.repeat(X.shape[0], 1)
        # mask out nans in input
        nanmask = torch.isnan(X)
        B[nanmask] = 0.
        X[nanmask] = 0.
        # concatenate X and B and acquire ability estimates;
        # again, we use only the mean for encoding
        Theta = self.q_theta(torch.cat((X, B), 1))[:, :self.num_concepts]
        return Theta.detach().numpy()

    def decode(self, Theta):
        """ Decodes the given knowledge into predicted
        test results.

        Parameters
        ----------
        Theta: ndarray
            A matrix with one row per student and one column per
            skill, where Theta[i, k] represents the estimated
            knowledge of student i for skill k.

        Returns
        -------
        Y: ndarray
            A matrix of predicted test responses for each student
            on each item.

        """
        Theta = torch.tensor(Theta, dtype=torch.float)
        # apply decoding layer to get the relevant knowledge for
        # each item
        Theta_hat = self.p_theta(Theta)
        # subtract the difficulties
        # use the item distribution mean as difficulty
        b = self.q_b.weight[:, 0].detach().numpy()
        # repeat for every student
        Y = Theta_hat.detach().numpy() - np.expand_dims(b, 0)
        # binarize result
        Y[Y <= 0.] = 0.
        Y[Y > 0.]  = 1.
        return Y


    def decode_proba(self, Theta):
        """ Decodes the given knowledge into success probabilities.

        Parameters
        ----------
        Theta: ndarray
            A matrix with one row per student and one column per
            skill, where Theta[i, k] represents the estimated
            knowledge of student i for skill k.

        Returns
        -------
        P: ndarray
            A matrix of predicted success probabilities for each
            student on each item.

        """
        Theta = torch.tensor(Theta, dtype=torch.float)
        # apply decoding layer to get the relevant knowledge for
        # each item
        Theta_hat = self.p_theta(Theta)
        # subtract the difficulties
        # use the item distribution mean as difficulty
        b = self.q_b.weight[:, 0].detach().numpy()
        # repeat for every student
        Y = Theta_hat.detach().numpy() - np.expand_dims(b, 0)
        # apply logistic function
        return 1. / (1. + np.exp(-Y))


    def predict(self, X):
        """ Auto-encodes the given test results.

        Parameters
        ----------
        X: ndarray
            A matrix of test responses where each row represents
            a student and each column represents a question.

        Returns
        -------
        Y: ndarray
            A matrix of predicted test responses for each student
            on each item.

        """
        Theta = self.encode(X)
        return self.decode(Theta)


    def predict_proba(self, X):
        """ Auto-encodes the given test results.

        Parameters
        ----------
        X: ndarray
            A matrix of test responses where each row represents
            a student and each column represents a question.

        Returns
        -------
        P: ndarray
            A matrix of predicted success probabilities for each
            student on each item.

        """
        Theta = self.encode(X)
        return self.decode_proba(Theta)


    def compute_loss(self, X):
        """ Computes the VIBO loss for the given responses.

        Parameters
        ----------
        X: class torch.Tensor
            A matrix of test responses where each row represents
            a student and each column represents a question.

        Returns
        -------
        loss: class torch.tensor
            The VIBO loss.

        """
        # sample item difficulties first
        mu_b     = self.q_b.weight[:, 0]
        logvar_b = self.q_b.weight[:, 1]
        std_b    = torch.exp(0.5 * logvar_b)
        b        = torch.randn_like(mu_b) * std_b + mu_b

        # repeat for every student
        B = b.repeat(X.shape[0], 1)
        # mask out nans in input
        X = X.clone().detach()
        nanmask = torch.isnan(X)
        B[nanmask] = 0.
        X[nanmask] = 0.

        # sample abilities next
        MuLogvar     = self.q_theta(torch.cat((X, B), 1))
        Mu_theta     = MuLogvar[:, :self.num_concepts]
        Logvar_theta = MuLogvar[:, self.num_concepts:]
        Std_theta    = torch.exp(0.5 * Logvar_theta)
        Theta        = torch.randn_like(Mu_theta) * Std_theta + Mu_theta

        # Compute logits for each response probability
        Logits       = self.p_theta(Theta) - B
        # mask out nans
        Logits[nanmask] = -100.

        # compute VIBO. First, we compute the binary crossentropy loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Logits, X)

        # add regularization/KL divergences
        if self.regul > 0.:
            loss = loss + .5 * self.regul * torch.mean(torch.square(mu_b) + torch.square(std_b) - logvar_b - 1.) + .5 * self.regul * torch.mean(torch.square(Mu_theta) + torch.square(Std_theta) - Logvar_theta - 1.)
        return loss


    def fit(self, X, Y = None, print_step = 0):
        """ Fits a model to the given response data.

        Parameters
        ----------
        X: ndarray
            A matrix of test responses where each row represents
            a student and each column represents a question.
        Y: ndarray (default = None)
            Not needed. Only here for consistency with sklearn
            interface.

        Returns
        -------
        self

        """
        # check that the dimensions fit with the training data
        if X.shape[1] != self.num_items:
            raise ValueError('Expected one column in X for each item.')

        X = torch.tensor(X, dtype=torch.float)

        # set up ADAM optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # start training
        for epoch in range(self.num_epochs):
            # set up a random permutation of the data
            perm = torch.randperm(X.shape[0])
            # iterate over all data as minibatches
            for k in range(0, X.shape[0], self.minibatch_size):
                optimizer.zero_grad()
                # sample a minibatch of data
                minibatch   = perm[k:k+self.minibatch_size]
                X_minibatch = X[minibatch, :]
                # compute the current loss for it
                loss = self.compute_loss(X_minibatch)
                # compute gradient
                loss.backward()
                # perform an optimizer step
                optimizer.step()
            # print current state if so requested
            if print_step > 0 and (epoch+1) % print_step == 0:
                print('loss after %d epochs: %g' % (epoch+1, loss.item()))

        return self

    def Q(self):
        return self.p_theta.weight.detach().numpy()

    def difficulties(self):
        return self.q_b.weight[:, 0].detach().numpy()
