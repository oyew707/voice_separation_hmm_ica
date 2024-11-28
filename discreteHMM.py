"""
-------------------------------------------------------
Implementation of a discrete Hidden Markov Model (HMM).
Source from DS-GA 1018 Time series course, NYU Center for Data Science
-------------------------------------------------------
Author:  Sebastian Wagner, einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "11/27/24"
-------------------------------------------------------
"""

# Imports
import numpy as np
import tensorflow as tf
from ICA import ICA
from typing import Tuple

# Constants

# Our class of interest.
class DiscreteHMM:
    """
    Class that implements a discrete HMM.

    Args:
        pi: Initial guess for the vector pi.
        transition_matrix: Initial guess for the transition matrix.
    """
    def __init__(self, pi: tf.Tensor, transition_matrix: tf.Tensor):
        """Initialize our class."""
        # Save the initial guess.
        self.pi = pi
        self.transition_matrix = transition_matrix

        # Some useful variables.
        self.dim_z = len(self.pi)

        # # observation_probability function
        # self.ica_model = ICA(k = self.dim_z, m = 2)
        # self.observation_probability = lambda state, x: self.ica_model.compute_likelihood(x, state)

    def calc_alpha_hat(self, observations: tf.Tensor, observation_probability: callable) -> Tuple[tf.Tensor, tf.Tensor]:
        """Given the observations, calculate the normalized forward pass.

        Args:
            observations: Observations for each time step.
            observation_probability: Function that returns the p(x_t|z_t)
        Returns:
            Normalized forward pass for each time step and latent state and
            normalization constants.
        """
        # Initialize a placeholder.
        alpha_hat = tf.zeros((len(observations) + 1, self.dim_z))
        c_t = tf.zeros((len(observations) + 1))

        # t=0 value. Consider c_0 = 1 for simplicity.
        alpha_hat[0] = self.pi
        c_t[0] = 1

        # Recurse forward.
        for t in range(1, len(alpha_hat)):
            for i in range(self.dim_z):
                for j in range(self.dim_z):
                    p_x_given_z = observation_probability(i, observations[t-1])
                    alpha_hat[t,i] += p_x_given_z * alpha_hat[t-1,j] * self.transition_matrix[j,i]
            # Don't forget to normalize alpha_hat!
            c_t[t] = alpha_hat[t,:].sum()
            alpha_hat[t] /= c_t[t]

        return alpha_hat, c_t

    def calc_beta_hat(self, observations: tf.Tensor, observation_probability: callable) -> tf.Tensor:
        """Given the observations, calculate the normalized backward pass.

        Args:
            observations: Observations for each time step.
            observation_probability: Function that returns the p(x_t|z_t)
        Returns:
            Normalized backward pass for each time step.
        """
        # Initialize a placeholder.
        beta_hat = tf.zeros((len(observations) + 1, self.dim_z))

        # Get the c_t values
        _, c_t = self.calc_alpha_hat(observations)

        # t=0 value.
        beta_hat[-1] = 1

        # Recurse backward.
        for t in range(len(beta_hat)-2, -1, -1):
            for i in range(self.dim_z):
                for j in range(self.dim_z):
                    p_x_given_Z = observation_probability(j, observations[t])
                    beta_hat[t,i] += beta_hat[t+1,j] * p_x_given_Z * self.transition_matrix[i,j]
            # Don't forget to normalize beta_hat!
            beta_hat[t] /= c_t[t+1]

        return beta_hat

    def p_zt_xt(self, observations: tf.Tensor, alpha_hat: tf.Tensor,
                beta_hat: tf.Tensor, c_t: tf.Tensor):
        """Calculate p(z_t|x_{1:t}) for all t.

        Args:
            observations: Observations for each time step.
            alpha_hat: Normalized forward pass output.
            beta_hat: Normalized backward pass output.
            c_t: Normalization constants for forward and backward pass.

        Returns:
            Value of p(z_t|x_{1:t}) for each time step and latent state index.

        Notes:
            You may not need all of the inputs.
        """
        return alpha_hat

    def p_zt_xT(self, observations: tf.Tensor, alpha_hat: tf.Tensor,
                beta_hat: tf.Tensor, c_t: tf.Tensor):
        """Calculate p(z_t|x_{1:T}) for all t.

        Args:
            observations: Observations for each time step.
            alpha_hat: Normalized forward pass output.
            beta_hat: Normalized backward pass output.
            c_t: Normalization constants for forward and backward pass.

        Returns:
            Value of p(z_t|x_{1:T}) for each time step and latent state index.

        Notes:
            You may not need all of the inputs.
        """

        return alpha_hat * beta_hat

    def log_p_sequence_xt(self, observations: tf.Tensor, latent_sequence: tf.Tensor, observation_probability:callable) -> float:
        """Log likelihood of the sequence given the data p(z_{1:T}|x_{1:T}).

        Args:
            observations: Observations for each time step.
            latent_sequence: Proposed latent sequence. Can be a probability
                distribution at each time step, in which case argmax will be taken
                to determine proposal.
            observation_probability: Function that returns the p(x_t|z_t)

        Returns:
            Log of the probability of the sequence given the data.
        """
        log_probability = 0
        sequence_indices = tf.math.argmax(latent_sequence, axis=-1)

        # Log probability for first time step.
        log_probability += tf.math.log(self.pi[sequence_indices[0]])

        # Log probability for remaining time steps.
        for t in range(len(observations)):
            log_probability += tf.math.log(self.transition_matrix[sequence_indices[t], sequence_indices[t+1]]) + \
                tf.math.log(observation_probability(sequence_indices[t], observations[t]))

        return log_probability
