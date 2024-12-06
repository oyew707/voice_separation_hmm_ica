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
import tensorflow as tf
from typing import Tuple


# Constants
LOG_EPS = 1e-10

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
        self.dim_z = self.pi.shape[0]

        # # observation_probability function
        # self.ica_model = ICA(k = self.dim_z, m = 2)
        # self.observation_probability = lambda state, x: self.ica_model.compute_likelihood(x, state)

    @tf.function
    def calc_obs_prob(self, observations: tf.Tensor, observation_probability: callable) -> tf.Tensor:
        """Given the observations, compute all observation probability (used in calc_alpha_hat and calc_beta_hat).

        Args:
            observations: Observations for each time step.
            observation_probability: Function that returns the p(x_t|z_t)
        Returns:
            observation probability
        """
        obs_probs = tf.stack([observation_probability(k, observations) for k in range(self.dim_z)])
        obs_probs = tf.transpose(obs_probs)
        return obs_probs

    @tf.function
    def calc_alpha_hat(self, observations: tf.Tensor, obs_probs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Given the observations, calculate the normalized forward pass.

        Args:
            observations: Observations for each time step.
            obs_probs: observation probabilities i.e. p(x_t|z_t)
        Returns:
            Normalized forward pass for each time step and latent state and
            normalization constants.
        """
        # Initialize a placeholder.
        T = observations.shape[0]
        alpha_hat = tf.TensorArray(tf.float32, size=T + 1, clear_after_read=False)
        c_t = tf.TensorArray(tf.float32, size=T + 1, clear_after_read=False)

        # t=0 value. Consider c_0 = 1 for simplicity.
        alpha_hat = alpha_hat.write(0, self.pi)
        c_t = c_t.write(0, tf.constant(1.0, dtype=tf.float32))

        # Recurse forward.
        for t in range(1, observations.shape[0] + 1):
            alpha_t1 = alpha_hat.read(t - 1)[:, tf.newaxis]
            new_alpha = tf.reduce_sum(
                alpha_t1 * self.transition_matrix, axis=0
            ) * obs_probs[t - 1]
            new_c = tf.reduce_sum(new_alpha) + LOG_EPS
            new_alpha /= new_c
            c_t = c_t.write(t, new_c)
            alpha_hat = alpha_hat.write(t, new_alpha)

        return alpha_hat.stack(), c_t.stack()

    @tf.function
    def calc_beta_hat(self, observations: tf.Tensor, obs_probs: tf.Tensor, c_t: tf.Tensor) -> tf.Tensor:
        """Given the observations, calculate the normalized backward pass.

        Args:
            observations: Observations for each time step.
            obs_probs: observation probabilities i.e. p(x_t|z_t)
            c_t: Normalization constants for forward pass.
        Returns:
            Normalized backward pass for each time step.
        """
        # Initialize a placeholder.
        beta_hat = tf.TensorArray(tf.float32, size=observations.shape[0] + 1, clear_after_read=False)

        # t=0 value.
        beta_hat = beta_hat.write(observations.shape[0], tf.ones(self.dim_z))

        # Recurse backward.
        for t in range(observations.shape[0] - 1, -1, -1):
            curr_beta = tf.reduce_sum(
                self.transition_matrix * obs_probs[t] *
                beta_hat.read(t + 1)[tf.newaxis, :], axis=1
            )
            curr_beta = curr_beta / c_t[t + 1]
            beta_hat = beta_hat.write(t, curr_beta)

        return beta_hat.stack()

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

    @tf.function
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

    @tf.function
    def p_zt_zt1_xt(self, observations: tf.Tensor, alpha_hat: tf.Tensor, beta_hat: tf.Tensor, c_t: tf.Tensor,
                    observation_probability: callable) -> tf.Tensor:
        """
        Calculate p(z_t, z_{t-1}|x_{1:T}).
        Args:
            observations: Observations for each time step.
            alpha_hat: Normalized forward pass output.
            beta_hat: Normalized backward pass output.
            c_t: Normalization constants for forward and backward pass.
            observation_probability: Function that returns the p(x_t|z_t)

        Returns:
            Value of p(z_t, z_{t-1}|x_{1:T}) for each time step and latent state index.
        """
        obs_probs = self.calc_obs_prob(observations, observation_probability)

        # Initialize result tensor
        T = observations.shape[0]
        p_zt_zt1_xt = tf.zeros((T, self.dim_z, self.dim_z))

        # Expand dimensions for broadcasting
        alpha_expanded = alpha_hat[:-1, :, tf.newaxis]  # [T, dim_z, 1]
        beta_expanded = beta_hat[1:, tf.newaxis, :]  # [T, 1, dim_z]
        obs_expanded = obs_probs[:, tf.newaxis, :]  # [T, 1, dim_z]
        c_expanded = c_t[1:, tf.newaxis, tf.newaxis]  # [T, 1, 1]
        trans_expanded = tf.expand_dims(self.transition_matrix, axis=0)  # [1, dim_z, dim_z]

        # Compute joint probability in one vectorized operation
        joint_prob = alpha_expanded * trans_expanded * obs_expanded * beta_expanded / c_expanded

        # Update result tensor
        p_zt_zt1_xt = tf.tensor_scatter_nd_update(
            p_zt_zt1_xt,
            tf.reshape(tf.range(0, T), [-1, 1]),
            joint_prob
        )

        return p_zt_zt1_xt

    def log_p_sequence_xt(self, observations: tf.Tensor, latent_sequence: tf.Tensor,
                          observation_probability: callable) -> float:
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
            log_probability += tf.math.log(self.transition_matrix[sequence_indices[t], sequence_indices[t + 1]]) + \
                               tf.math.log(observation_probability(sequence_indices[t], observations[t]))

        return log_probability
