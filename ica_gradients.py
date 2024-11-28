"""
-------------------------------------------------------
Gradient computations for Hidden Markov Independent Component Analysis
Supports both analytical gradients and automatic differentiation
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "11/27/24"
-------------------------------------------------------
"""

# Imports
import tensorflow as tf
from scipy.special import digamma
from typing import Tuple, Optional
from ICA import ICA


# Constants

class ICAGradients:
    """
    -------------------------------------------------------
    Computes gradients for HMICA model parameters using either analytical formulas
    or automatic differentiation
    -------------------------------------------------------
    Parameters:
        ica_model - The ICA model instance
        use_analytical - Whether to use analytical gradients (bool)
    ------------------------------------------------
    """

    def __init__(self, ica_model: ICA, use_analytical: bool = True):
        self.ica = ica_model
        self.use_analytical = use_analytical

    def compute_gradients(self, x: tf.Tensor, gamma_k: tf.Tensor,
                          state: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        """
        -------------------------------------------------------
        Computes gradients for all HMICA parameters using either method
        -------------------------------------------------------
        Parameters:
            x - Input mixed signals (tf.Tensor of shape [time_steps, x_dim])
            gamma_k - State responsibilities (tf.Tensor of shape [time_steps])
            state - Current state index (int)
        Returns:
            W_grad - Gradient for unmixing matrix
            R_grad - Gradient for shape parameters
            beta_grad - Gradient for scale parameters
            C_grad - Gradient for GAR coefficients (if using GAR)
        -------------------------------------------------------
        """
        if self.use_analytical:
            return self._compute_analytical_gradients(x, gamma_k, state)
        else:
            return self._compute_auto_gradients(x, gamma_k, state)

    def _compute_analytical_gradients(self, x: tf.Tensor, gamma_k: tf.Tensor,
                                      state: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        """
        -------------------------------------------------------
        Computes gradients using derived analytical formulas
        -------------------------------------------------------
        Parameters:
            x - Input mixed signals (tf.Tensor of shape [time_steps, x_dim])
            gamma_k - State responsibilities (tf.Tensor of shape [time_steps])
            state - Current state index (int)
        Returns:
            Tuple of gradients (W, R, beta, C)
        -------------------------------------------------------
        """
        # Get current parameters and compute sources
        W = self.ica.W[state]
        sources = self.ica.get_sources(x, state)
        gamma_sum = tf.reduce_sum(gamma_k)

        # Unmixing matrix gradient
        W_pinv_T = tf.transpose(tf.linalg.pinv(W))

        # Initialize gradient matrix
        W_grad = tf.zeros_like(W)  # Shape: [x_dim, m]

        # Compute gradient elements according to paper formula
        for i in range(self.ica.x_dim):  # Iterate over input dimensions
            for j in range(self.ica.m):  # Iterate over sources
                R_j = self.ica.ge[state].R[j]
                beta_j = self.ica.ge[state].beta[j]

                # Compute source terms for a_j[t]
                abs_src = tf.abs(sources[:, j])
                power_term = tf.pow(abs_src, R_j - 1)
                sign_term = tf.sign(sources[:, j])

                # Compute y_j[t] = sum_i(W_ij * a_i[t])
                y_j = tf.reduce_sum(W[:, j] * sources, axis=1)

                # Compute gradient term
                grad_term = R_j * beta_j * sign_term * power_term * y_j * gamma_k
                W_grad[i, j] = W_pinv_T[i, j] - (1 / gamma_sum) * tf.reduce_sum(grad_term)

        # Beta gradient (optimal value)
        beta_grad = tf.zeros_like(self.ica.ge[state].beta)
        for i in range(self.ica.m):
            R_i = self.ica.ge[state].R[i]
            abs_src_R = tf.pow(tf.abs(sources[:, i]), R_i)
            weighted_sum = tf.reduce_sum(gamma_k * abs_src_R)
            beta_grad = gamma_sum / (R_i * weighted_sum)

        # R gradient
        R_grad = tf.zeros_like(self.ica.ge[state].R)
        for i in range(self.ica.m):
            R_i = self.ica.ge[state].R[i]
            beta_i = self.ica.ge[state].beta[i]

            # Digamma term = φ(1/R_i )=(Γ^′ (1/R_i ))/Γ(1/Ri )
            digamma_term = digamma(1 / R_i)

            # Log sum term
            abs_src = tf.abs(sources[:, i])
            log_abs = tf.math.log(abs_src)
            power_term = tf.pow(abs_src, R_i)
            sum_term = tf.reduce_sum(gamma_k * power_term * log_abs)

            R_grad[i] = (1 / R_i) + (1 / (R_i ** 2)) * (tf.math.log(beta_i) + digamma_term) - \
                        (beta_i / gamma_sum) * sum_term

        # GAR coefficients gradient if using GAR
        if self.ica.use_gar:
            C_grad = self._compute_gar_gradients(sources, gamma_k, state)
        else:
            C_grad = None

        return W_grad, R_grad, beta_grad, C_grad

    def _compute_gar_gradients(self, sources: tf.Tensor, gamma_k: tf.Tensor,
                               state: int) -> tf.Tensor:
        """
        -------------------------------------------------------
        Computes gradients for GAR coefficients analytically
        -------------------------------------------------------
        Parameters:
            sources - Source signals (tf.Tensor of shape [time_steps, x_dim])
            gamma_k - State responsibilities (tf.Tensor of shape [time_steps])
            state - Current state index (int)
        Returns:
            C_grad - Gradient for GAR coefficients
        -------------------------------------------------------
        """
        gamma_sum = tf.reduce_sum(gamma_k)
        C_grad = tf.zeros_like(self.ica.gar.C[state])

        # Get prediction errors
        errors = self.ica.gar.compute_prediction_error(sources, state)

        for i in range(self.ica.m):
            R_i = self.ica.ge[state].R[i]

            for d in range(self.ica.gar.p):
                # Get delayed source values
                delayed_source = sources[self.ica.gar.p - d - 1:-d - 1, i]

                # Error term computation
                error_term = R_i * tf.sign(errors[:, i]) * tf.pow(tf.abs(errors[:, i]), R_i - 1)

                # Weighted sum
                C_grad[i, d] = (1 / gamma_sum) * tf.reduce_sum(gamma_k * delayed_source * error_term)

        return C_grad

    def _compute_auto_gradients(self, x: tf.Tensor, gamma_k: tf.Tensor,
                                state: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        """
        -------------------------------------------------------
        Computes gradients using TensorFlow's automatic differentiation
        -------------------------------------------------------
        Parameters:
            x - Input mixed signals (tf.Tensor of shape [time_steps, x_dim])
            gamma_k - State responsibilities (tf.Tensor of shape [time_steps])
            state - Current state index (int)
        Returns:
            Tuple of gradients (W, R, beta, C)
        -------------------------------------------------------
        """
        watched_vars = [
            self.ica.W[state],
            self.ica.ge[state].R,
            self.ica.ge[state].beta
        ]

        if self.ica.use_gar:
            watched_vars.append(self.ica.gar.C[state])

        # disable automatic tracking, fine-grained control over which variables are watched
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for var in watched_vars:
                tape.watch(var)

            log_likelihood = self.ica.compute_log_likelihood(x, state)
            weighted_ll = tf.reduce_sum(gamma_k * log_likelihood)

        grads = tape.gradient(weighted_ll, watched_vars)

        W_grad, R_grad, beta_grad = grads[:3]
        C_grad = grads[3] if self.ica.use_gar else None

        return W_grad, R_grad, beta_grad, C_grad
