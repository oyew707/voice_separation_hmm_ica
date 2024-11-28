"""
-------------------------------------------------------
generalized AutoRegressive (AR) model
-------------------------------------------------------
Author:  einsteinoyewole
ID:      [your ID]
Email:   [your email address]
__updated__ = "11/27/24"
-------------------------------------------------------
"""

# Imports
import tensorflow as tf
import numpy as np

# Constants

class GeneralizedAutoRegressive:
    """
    -------------------------------------------------------
    A generalized autoregressive model that predicts future values
    based on p previous values for k different sources
    -------------------------------------------------------
    Parameters:
        k - Number of states in the Hidden Markov Model (int > 0)
        m - Number of independent sources to model (int > 0)
        p - Order of the autoregressive model (int > 0)
    -------------------------------------------------------
    """

    def __init__(self, k: int, m: int, p: int):
        assert m > 0, "Number of sources must be a positive integer"
        assert p > 0, "Order of AR model must be a positive integer"
        assert k > 0, "Number of states must be a positive integer"

        self.m = m  # Number of sources
        self.p = p  # Order of the AR model
        self.k = k  # Number of states

        # Initialize AR coefficients using Xavier/Glorot initialization
        limit = np.sqrt(6 / (m + p))
        self.C = [tf.Variable(
            tf.random.uniform(shape=( m, p), minval=-limit, maxval=limit),
            dtype=tf.float32,
            name=f'ar_coefficients_{i}'
        ) for i in range(k)]

    @tf.function
    def predict(self, A: tf.Tensor, state: int) -> tf.Tensor:
        """
        -------------------------------------------------------
        Predicts future values for each source using the AR coefficients.
        Uses the formula: â_i[t] = -∑(c_i[d] * a_i[t-d]) for d=1 to p and i=1 to k
        -------------------------------------------------------
        Parameters:
            A - Source signals tensor (tf.Tensor of shape [time_steps, m])
            state - Current state index (int, 0 <= state < k)
        Returns:
            predictions - Predicted values for each source (tf.Tensor of shape
                        [time_steps-p, m])
        -------------------------------------------------------
        """
        assert 0 <= state < self.k, "Invalid state index"
        # Create frames for each source
        windows = tf.signal.frame(A, self.p, 1)  # Shape: [time_steps-p+1, p, k]

        # Transpose to get shape [time_steps-p+1, k, p] to align with coefficients
        windows = tf.transpose(windows, [0, 2, 1])

        # Flip the windows to get the correct time ordering (most recent last)
        windows = tf.reverse(windows, axis=[2])

        # Get coefficients for current state
        state_coeffs = self.C[state]  # Shape: [m, p]

        # Compute predictions
        predictions = -tf.reduce_sum(windows * state_coeffs, axis=2)

        return predictions

    def compute_prediction_error(self, A: tf.Tensor, state: int) -> tf.Tensor:
        """
        -------------------------------------------------------
        Computes the prediction errors (residuals) between actual and
        predicted values for each source
        -------------------------------------------------------
        Parameters:
            A - Source signals tensor (tf.Tensor of shape [time_steps, m])
            state - Current state index (int, 0 <= state < k)
        Returns:
            errors - Prediction errors for each source (tf.Tensor of shape
                    [time_steps-p, m])
        -------------------------------------------------------
        """
        predictions = self.predict(A, state)
        # actual_values = A[self.p:, :]
        return A - predictions
