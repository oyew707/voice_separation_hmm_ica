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
        use_analytical - Whether to use analytical gradients (bool)
    ------------------------------------------------
    """

    def __init__(self, use_analytical: bool = True, **kwargs):
        self.use_analytical = use_analytical

    def compute_gradients(self, x: tf.Tensor, gamma_k: tf.Tensor,
                          state: int, ica_model: ICA) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        """
        -------------------------------------------------------
        Computes gradients for all HMICA parameters using either method
        -------------------------------------------------------
        Parameters:
            x - Input mixed signals (tf.Tensor of shape [time_steps, x_dim])
            gamma_k - State responsibilities (tf.Tensor of shape [time_steps])
            state - Current state index (int)
            ica_model - The ICA model instance
        Returns:
            W_grad - Gradient for unmixing matrix
            R_grad - Gradient for shape parameters
            beta_grad - Gradient for scale parameters
            C_grad - Gradient for GAR coefficients (if using GAR)
        -------------------------------------------------------
        """
        if self.use_analytical:
            return self._compute_analytical_gradients(x, gamma_k, state, ica_model)
        else:
            return self._compute_auto_gradients(x, gamma_k, state, ica_model)

    def _compute_analytical_gradients(self, x: tf.Tensor, gamma_k: tf.Tensor,
                                      state: int, ica_model: ICA) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        """
        -------------------------------------------------------
        Computes gradients using derived analytical formulas
        -------------------------------------------------------
        Parameters:
            x - Input mixed signals (tf.Tensor of shape [time_steps, x_dim])
            gamma_k - State responsibilities (tf.Tensor of shape [time_steps])
            state - Current state index (int)
            ica_model - The ICA model instance
        Returns:
            Tuple of gradients (W, R, beta, C)
        -------------------------------------------------------
        """
        # Get current parameters and compute sources
        W = ica_model.W[state]
        sources = ica_model.get_sources(x, state)
        gamma_sum = tf.reduce_sum(gamma_k)

        # Unmixing matrix gradient
        W_pinv_T = tf.transpose(tf.linalg.pinv(W))

        # Initialize gradient matrix
        W_grad = tf.zeros_like(W)  # Shape: [x_dim, m]

        # Compute gradient elements according to paper formula
        for i in range(ica_model.x_dim):  # Iterate over input dimensions
            for j in range(ica_model.m):  # Iterate over sources
                R_j = ica_model.ge[state].R[j]
                beta_j = ica_model.ge[state].beta[j]

                # Compute source terms for a_j[t]
                abs_src = tf.abs(sources[:, j])
                power_term = tf.pow(abs_src, R_j - 1)
                sign_term = tf.sign(sources[:, j])

                # Compute y_j[t] = sum_i(W_ij * a_i[t])
                y_j = tf.reduce_sum(W[:, j] * sources, axis=1)

                # Compute gradient term
                grad_term = R_j * beta_j * sign_term * power_term * y_j * gamma_k
                upd = W_pinv_T[i, j] - (1 / gamma_sum) * tf.reduce_sum(grad_term)
                W_grad = tf.tensor_scatter_nd_update(W_grad, [[i, j]], [upd])

        # Beta gradient (optimal value)
        beta_grad = tf.zeros_like(ica_model.ge[state].beta)
        for i in range(ica_model.m):
            R_i = ica_model.ge[state].R[i]
            abs_src_R = tf.pow(tf.abs(sources[:, i]), R_i)
            weighted_sum = tf.reduce_sum(gamma_k * abs_src_R)
            beta_grad = tf.tensor_scatter_nd_update(beta_grad, [[i]], [gamma_sum / (R_i * weighted_sum)])

        # R gradient
        R_grad = tf.zeros_like(ica_model.ge[state].R)
        for i in range(ica_model.m):
            R_i = ica_model.ge[state].R[i]
            beta_i = ica_model.ge[state].beta[i]

            # Digamma term = φ(1/R_i )=(Γ^′ (1/R_i ))/Γ(1/Ri )
            digamma_term = tf.math.digamma(1 / R_i)

            # Log sum term
            abs_src = tf.abs(sources[:, i])
            log_abs = tf.math.log(abs_src)
            power_term = tf.pow(abs_src, R_i)
            sum_term = tf.reduce_sum(gamma_k * power_term * log_abs)

            upd = (1 / R_i) + (1 / (R_i ** 2)) * (tf.math.log(beta_i) + digamma_term) - \
                  (beta_i / gamma_sum) * sum_term
            R_grad = tf.tensor_scatter_nd_update(R_grad, [[i]], [upd])

        # GAR coefficients gradient if using GAR
        if self.ica.use_gar:
            C_grad = self._compute_gar_gradients(sources, gamma_k, state)
        else:
            C_grad = None

        return W_grad, R_grad, beta_grad, C_grad

    @staticmethod
    def _compute_gar_gradients(sources: tf.Tensor, gamma_k: tf.Tensor,
                               state: int, ica_model: ICA) -> tf.Tensor:
        """
        -------------------------------------------------------
        Computes gradients for GAR coefficients analytically
        -------------------------------------------------------
        Parameters:
            sources - Source signals (tf.Tensor of shape [time_steps, x_dim])
            gamma_k - State responsibilities (tf.Tensor of shape [time_steps])
            state - Current state index (int)
            ica_model - The ICA model instance
        Returns:
            C_grad - Gradient for GAR coefficients
        -------------------------------------------------------
        """
        gamma_sum = tf.reduce_sum(gamma_k)
        C_grad = tf.zeros_like(ica_model.gar.C[state])

        # Get prediction errors
        errors = ica_model.gar.compute_prediction_error(sources, state)

        for i in range(ica_model.m):
            R_i = ica_model.ge[state].R[i]

            for d in range(ica_model.gar.p):
                # Get delayed source values
                delayed_source = sources[:, i]  # sources[self.ica.gar.p - d - 1:-d - 1, i]

                # Error term computation
                error_term = R_i * tf.sign(errors[:, i]) * tf.pow(tf.abs(errors[:, i]), R_i - 1)

                # Weighted sum
                upd = (1 / gamma_sum) * tf.reduce_sum(gamma_k * delayed_source * error_term)
                C_grad = tf.tensor_scatter_nd_update(C_grad, [[i, d]], [upd])

        return C_grad

    @staticmethod
    def _compute_auto_gradients(x: tf.Tensor, gamma_k: tf.Tensor,
                                state: int, ica_model: ICA) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        """
        -------------------------------------------------------
        Computes gradients using TensorFlow's automatic differentiation
        -------------------------------------------------------
        Parameters:
            x - Input mixed signals (tf.Tensor of shape [time_steps, x_dim])
            gamma_k - State responsibilities (tf.Tensor of shape [time_steps])
            state - Current state index (int)
            ica_model - The ICA model instance (ICA)
        Returns:
            Tuple of gradients (W, R, beta, C)
        -------------------------------------------------------
        """
        watched_vars = [
            ica_model.W[state],
            ica_model.ge[state].R,
            ica_model.ge[state].beta
        ]

        if ica_model.use_gar:
            watched_vars.append(ica_model.gar.C[state])

        # disable automatic tracking, fine-grained control over which variables are watched
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for var in watched_vars:
                tape.watch(var)

            log_likelihood = ica_model.compute_log_likelihood(x, state)
            weighted_ll = tf.reduce_sum(gamma_k * log_likelihood)

        grads = tape.gradient(weighted_ll, watched_vars)

        W_grad, R_grad, beta_grad = grads[:3]
        C_grad = grads[3] if ica_model.use_gar else None

        return W_grad, R_grad, beta_grad, C_grad
