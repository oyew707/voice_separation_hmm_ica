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
RANDOM_SEED = 2
tf.random.set_seed(RANDOM_SEED)


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

        # Add small epsilon to prevent zero probabilities
        obs_probs = obs_probs + 1e-12

        # This keeps the relative likelihoods between states while making numbers more manageable
        normalizer = tf.reduce_sum(obs_probs, axis=1, keepdims=True)
        obs_probs = obs_probs / normalizer
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

    @tf.function
    def infer_states_forward_backward(self, observations: tf.Tensor, observation_probability: callable) -> float:
        """the sequence of states given the data p(z_{1:T}|x_{1:T}).
            use the forward-backward algorithm (p(z_t|x_{1:T})) for state inference
        Args:
            observations: Observations for each time step.
            observation_probability: Function that returns the p(x_t|z_t)

        Returns:
            One-hot encoding of state sequence.
        """
        # Pre-compute all observation probabilities at once [T, dim_z]
        obs_probs = self.calc_obs_prob(observations, observation_probability)

        # Calculate alpha_hat and c_t
        alpha_hat, c_t = self.calc_alpha_hat(observations, obs_probs)

        # calculate beta_hat
        beta_hat = self.calc_beta_hat(observations, obs_probs, c_t)

        # calculate p(z_t|x_{1:T})
        posterior = self.p_zt_xT(observations, alpha_hat, beta_hat, c_t)
        # Normalize posterior probabilities
        posterior = posterior / tf.reduce_sum(posterior, axis=1, keepdims=True)
        sequence_indices = tf.math.argmax(posterior, axis=-1)

        map_sequence = tf.one_hot(sequence_indices, self.dim_z)

        return map_sequence

    def viterbi_for_inference(self, observations: tf.Tensor, observation_probability: callable) -> Tuple[
        tf.Tensor, tf.Tensor]:
        """Run the Viterbi algorithm on our dHMM observations.

        Args:
            observations: Observations for each time step.
            observation_probability: Function that returns the p(x_t|z_t)
        Returns:
            One-hot encoding of MAP sequence.
        """
        # Initialize variables.
        T = observations.shape[0]

        # Pre-compute all observation probabilities at once [T, dim_z]
        obs_probs = self.calc_obs_prob(observations, observation_probability)
        log_obs_probs = tf.math.log(obs_probs + LOG_EPS)

        # Initialize weight matrix w[t,k] and backpointer matrix m[t,k]
        # w has T+1 rows because we need initial weights w₀
        w_array = tf.TensorArray(tf.float32, size=T + 1, clear_after_read=False)
        m_array = tf.TensorArray(tf.int32, size=T, clear_after_read=False)

        # Initialize w₀ with log of initial probabilities
        w_array = w_array.write(0, tf.math.log(self.pi + LOG_EPS) + log_obs_probs[0])

        # Pre-compute log transition probabilities
        log_trans = tf.math.log(self.transition_matrix + LOG_EPS)

        # Forward pass to fill w and m matrices
        for t in tf.range(1, T):
            # Calculate all possible paths to each state
            # [dim_z, 1] + [dim_z, dim_z] -> broadcasting to [dim_z, dim_z]
            all_paths = tf.expand_dims(w_array.read(t - 1), 1) + log_trans

            # Find best previous state and weight for all current states at once
            max_weights = tf.reduce_max(all_paths, axis=0) + log_obs_probs[t]  # [dim_z]
            best_prev_states = tf.cast(tf.argmax(all_paths, axis=0), tf.int32)  # [dim_z]

            # Update matrices for all states at once
            w_array = w_array.write(t, max_weights)
            m_array = m_array.write(t - 1, best_prev_states)

        w_matrix, m_matrix = w_array.stack(), m_array.stack()

        # Backtrack to find MAP sequence
        map_path = tf.TensorArray(tf.int32, size=T, clear_after_read=False)

        # Get z_T
        last_state = tf.cast(tf.argmax(w_matrix[T - 1]), tf.int32)
        map_path = map_path.write(T - 1, last_state)

        # Backtrack
        current_state = last_state
        for t in tf.range(T - 2, -1, -1):
            current_state = m_matrix[t, current_state]
            map_path = map_path.write(t, current_state)

        # Stack final path and convert to one-hot
        map_path = map_path.stack()
        map_sequence = tf.one_hot(map_path, self.dim_z)

        return map_sequence, w_matrix
