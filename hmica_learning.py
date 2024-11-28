"""
-------------------------------------------------------
Hidden Markov Independent Component Analysis (HMICA) Learning Algorithm
Implements the learning procedure described in Penny et al. (2000)
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "11/28/24"
-------------------------------------------------------
"""


# Imports
import tensorflow as tf
from typing import Tuple, Optional
from discreteHMM import DiscreteHMM
from ICA import ICA
from ica_gradients import ICAGradients

# Constants


class HMICALearner:
    """
    -------------------------------------------------------
    Implements the learning algorithm for Hidden Markov Independent
    Component Analysis as described in the paper.
    -------------------------------------------------------
    Parameters:
        k - Number of states in HMM (int > 0)
        m - Number of sources to extract (int > 0)
        x_dims - Number of observed dimensions (int > 0)
        use_gar - Whether to use Generalized Autoregressive modeling (bool)
        gar_order - Order of GAR model if used (int > 0)
        update_interval - Number of ICA iterations before updating GAR (int > 0)
    -------------------------------------------------------
    """

    def __init__(self, k: int, m: int, x_dims: int, use_gar: bool = False,
                 gar_order: int = 1, update_interval: int = 10, learning_rates: Optional[dict] = None):
        # Initialize models
        self.k = k
        self.m = m
        self.x_dims = x_dims
        self.use_gar = use_gar
        self.update_interval = update_interval

        # Default learning rates
        default_rates = {
            'W': 0.01,  # Unmixing matrix
            'R': 0.01,  # Shape parameter
            'beta': 0.01,  # Scale parameter
            'C': 0.01  # GAR coefficients
        }
        self.learning_rates = learning_rates or default_rates

        # Initialize HMM with uniform initial probabilities and transitions
        pi = tf.ones(k) / k
        A = tf.ones((k, k)) / k
        self.hmm = DiscreteHMM(pi, A)

        # Initialize ICA model
        self.ica = ICA(k, m, x_dims, use_gar, gar_order)

        # Initialize gradient computer
        self.grad_computer = ICAGradients(self.ica)


    @staticmethod
    def compute_convergence(new_ll: float, old_ll: float, init_ll: float,
                            delta: float = 1e-4) -> bool:
        """
        -------------------------------------------------------
        Checks convergence criterion using log-likelihood values
        -------------------------------------------------------
        Parameters:
            new_ll - New log-likelihood value (float)
            old_ll - Previous log-likelihood value (float)
            init_ll - Initial log-likelihood value (float)
            delta - Convergence threshold (float > 0)
        Returns:
            converged - Whether convergence criterion is met (bool)
        -------------------------------------------------------
        """
        if init_ll == old_ll:
            return False
        return (new_ll - init_ll) / (old_ll - init_ll) < 1 + delta

    def train(self, x: tf.Tensor, max_iter: int = 100,
              hmm_tol: float = 1e-4, ica_tol: float = 1e-2) -> dict:
        """
        -------------------------------------------------------
        Trains the HMICA model using the EM algorithm
        -------------------------------------------------------
        Parameters:
            x - Input data (tf.tensor of shape [time_steps, x_dims])
            max_iter - Maximum number of EM iterations (int > 0)
            hmm_tol - Convergence tolerance for HMM updates (float > 0)
            ica_tol - Convergence tolerance for ICA updates (float > 0)
        Returns:
            history - Dictionary containing training history
        -------------------------------------------------------
        """
        history = {'hmm_ll': [], 'ica_ll': []}

        # Initial likelihood computation
        init_hmm_ll = self._compute_total_likelihood(x)
        old_hmm_ll = init_hmm_ll

        # Main EM loop
        for iteration in range(max_iter):
            # E-step: Forward-backward algorithm
            obs_prob = lambda state, x: self.ica.compute_likelihood(x, state)
            alpha_hat, c_t = self.hmm.calc_alpha_hat(x, obs_prob)
            beta_hat = self.hmm.calc_beta_hat(x, obs_prob)
            g = self.hmm.p_zt_xT(x, alpha_hat, beta_hat, c_t)
            responsibilities = g / tf.reduce_sum(g, axis=1, keepdims=True)


            # M-step for each state
            for k in range(self.k):
                gamma_k = responsibilities[:, k]

                # Initialize ICA iteration tracking
                init_ica_ll = self._compute_ica_likelihood(x, k, gamma_k)
                old_ica_ll = init_ica_ll
                ica_iter = 0

                # ICA update loop
                while True and ica_iter < max_iter:
                    # Get sources and errors
                    sources = self.ica.get_sources(x, k)
                    if self.use_gar:
                        errors = self.ica.gar.compute_prediction_error(sources, k)

                    # Compute and apply gradients
                    W_grad, R_grad, beta_grad, C_grad = self.grad_computer.compute_gradients(
                        x, gamma_k, k)

                    # Update parameters using gradient steps
                    self._update_parameters(k, W_grad, R_grad, beta_grad, C_grad, ica_iter)

                    # Check ICA convergence
                    new_ica_ll = self._compute_ica_likelihood(x, k, gamma_k)
                    if self.compute_convergence(new_ica_ll, old_ica_ll,
                                                init_ica_ll, ica_tol):
                        break

                    old_ica_ll = new_ica_ll
                    ica_iter += 1

                history['ica_ll'].append(new_ica_ll)

            # Update HMM parameters
            self._update_hmm_parameters(responsibilities)

            # Check HMM convergence
            new_hmm_ll = self._compute_total_likelihood(x)
            if self.compute_convergence(new_hmm_ll, old_hmm_ll,
                                        init_hmm_ll, hmm_tol):
                break

            old_hmm_ll = new_hmm_ll
            history['hmm_ll'].append(new_hmm_ll)

        return history

    def _compute_ica_likelihood(self, x: tf.Tensor, state: int,
                                gamma_k: tf.Tensor) -> float:
        """
        -------------------------------------------------------
        Computes weighted log-likelihood for ICA model in given state
        -------------------------------------------------------
        Parameters:
            x - Input data (tf.Tensor of shape [time_steps, x_dims])
            state - Current state index (int)
            gamma_k - State responsibilities (tf.Tensor of shape [time_steps])
        Returns:
            ll - Weighted log-likelihood value (float)
        -------------------------------------------------------
        """
        log_probs = self.ica.compute_log_likelihood(x, state)
        return tf.reduce_sum(gamma_k * log_probs)

    def _update_parameters(self, state: int, W_grad: tf.Tensor,
                           R_grad: tf.Tensor, beta_grad: tf.Tensor,
                           C_grad: Optional[tf.Tensor], ica_iter: int):
        """
        -------------------------------------------------------
        Updates model parameters using optimizers created on the fly
        -------------------------------------------------------
        Parameters:
            state - Current state index (int)
            W_grad - Gradient for unmixing matrix (tf.Tensor)
            R_grad - Gradient for shape parameters (tf.Tensor)
            beta_grad - Gradient for scale parameters (tf.Tensor)
            C_grad - Gradient for GAR coefficients (Optional[tf.Tensor])
            ica_iter - Current ICA iteration (int)
        -------------------------------------------------------
        """
        # Update unmixing matrix
        W_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rates['W'])
        W_vars = [self.ica.W[state]]
        W_grads = [W_grad]
        W_optimizer.apply_gradients(zip(W_grads, W_vars))

        # Update source distribution parameters
        R_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rates['R'])
        R_vars = [self.ica.ge[state].R]
        R_grads = [R_grad]
        R_optimizer.apply_gradients(zip(R_grads, R_vars))

        beta_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rates['beta'])
        beta_vars = [self.ica.ge[state].beta]
        beta_grads = [beta_grad]
        beta_optimizer.apply_gradients(zip(beta_grads, beta_vars))

        # Update GAR coefficients if using GAR
        if self.use_gar and C_grad is not None and (ica_iter + 1) % self.update_interval == 0:
            C_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rates['C'])
            C_vars = [self.ica.gar.C[state]]
            C_grads = [C_grad]
            C_optimizer.apply_gradients(zip(C_grads, C_vars))

    def _update_hmm_parameters(self, responsibilities: tf.Tensor):
        """
        -------------------------------------------------------
        Updates HMM transition and initial probability parameters
        -------------------------------------------------------
        Parameters:
            responsibilities - State responsibilities (tf.tensor array of shape
                             [time_steps, k])
        -------------------------------------------------------
        """
        # Update initial state probabilities
        self.hmm.pi = responsibilities[0]

        # Update transition probabilities
        T = len(responsibilities)
        A = tf.zeros((self.k, self.k))

        for t in range(T - 1):
            A += tf.keras.ops.outer(responsibilities[t], responsibilities[t + 1])

        # Normalize
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.hmm.transition_matrix = A / row_sums

    def _compute_total_likelihood(self, x: tf.Tensor, responsibilities: tf.Tensor) -> float:
        """
        -------------------------------------------------------
        Computes complete joint log-likelihood of the HMICA model:
        L = ∑_(k=1)^K γ_k(0) log π_k +
            ∑_(t=1)^T ∑_(k'=1)^K ∑_(k=1)^K p(z_(t,k'),z_(t-1,k)|x_(1:T)) log A_(k,k') +
            ∑_(t=1)^T ∑_(k=1)^K γ_k(t) log p(x_t|z_t=k)
        -------------------------------------------------------
        Parameters:
            x - Input data (tf.Tensor of shape [time_steps, x_dims])
        Returns:
            ll - Complete joint log-likelihood value (float)
        -------------------------------------------------------
        """
        T = len(x)

        # 1. Initial state probabilities: ∑_(k=1)^K γ_k(0) log π_k
        initial_ll = tf.reduce_sum(
            responsibilities[0] * tf.math.log(self.hmm.pi)
        )

        # 2. Transition probabilities:
        # ∑_(t=1)^T ∑_(k'=1)^K ∑_(k=1)^K p(z_(t,k'),z_(t-1,k)|x_(1:T)) log A_(k,k')
        transition_ll = 0.0
        for t in range(1, T):
            # Compute joint probability of consecutive states
            joint_prob = self._compute_joint_state_prob(x, t, responsibilities)
            # Add to log-likelihood
            transition_ll += tf.reduce_sum(
                joint_prob * tf.math.log(self.hmm.transition_matrix)
            )

        # 3. ICA emission probabilities: ∑_(t=1)^T ∑_(k=1)^K γ_k(t) log p(x_t|z_t=k)
        emission_ll = 0.0
        for k in range(self.k):
            # Get log-likelihood for current state
            state_ll = self.ica.compute_log_likelihood(x, k)
            # Weight by state responsibilities
            emission_ll += tf.reduce_sum(responsibilities[:, k] * state_ll)

        return float(initial_ll + transition_ll + emission_ll)

    def _compute_joint_state_prob(self, x: tf.Tensor, t: int,
                                  responsibilities: tf.Tensor) -> tf.Tensor:
        """
        -------------------------------------------------------
        Computes joint probability p(z_t,z_{t-1}|x_{1:T}) using forward-backward
        variables and transition matrix
        -------------------------------------------------------
        Parameters:
            x - Input data (tf.Tensor of shape [time_steps, x_dims])
            t - Current time step (int)
            responsibilities - State responsibilities (tf.tensor of shape [T, K])
        Returns:
            joint_prob - Joint state probabilities (tf.Tensor of shape [K, K])
        -------------------------------------------------------
        """

        # Get forward-backward variables
        def obs_prob(state: int, xt: tf.Tensor) -> float:
            return float(self.ica.compute_likelihood(tf.expand_dims(xt, 0), state))

        alpha_hat, c_t = self.hmm.calc_alpha_hat(x, obs_prob)
        beta_hat = self.hmm.calc_beta_hat(x, obs_prob)

        # Compute joint probability for all state pairs
        joint_prob = tf.zeros((self.k, self.k))
        for k in range(self.k):
            for k_prime in range(self.k):
                # p(z_t=k',z_{t-1}=k|x_{1:T}) =
                # α_{t-1}(k) * A_{k,k'} * p(x_t|z_t=k') * β_t(k') / c_t
                joint_prob[k, k_prime] = (
                        alpha_hat[t - 1, k] *
                        self.hmm.transition_matrix[k, k_prime] *
                        obs_prob(k_prime, x[t]) *
                        beta_hat[t, k_prime] /
                        c_t[t]
                )

        return joint_prob
