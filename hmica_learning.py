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
from typing import Optional
from ICA import ICA
from discreteHMM import DiscreteHMM
from ica_gradients import ICAGradients


# Constants
tf.config.run_functions_eagerly(True)

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

        init_hmm_ll, new_hmm_ll = None, None

        # Main EM loop
        for iteration in range(max_iter):
            # E-step: Forward-backward algorithm
            obs_prob = lambda state, x: self.ica.compute_likelihood(x, state)
            alpha_hat, c_t = self.hmm.calc_alpha_hat(x, obs_prob)
            beta_hat = self.hmm.calc_beta_hat(x, obs_prob)
            g = self.hmm.p_zt_xT(x, alpha_hat, beta_hat, c_t)
            responsibilities = g / tf.reduce_sum(g, axis=1, keepdims=True)
            # Initial likelihood computation
            if init_hmm_ll is None:
                init_hmm_ll = self._compute_total_likelihood(x, responsibilities, alpha_hat, beta_hat, c_t)
                old_hmm_ll = init_hmm_ll
            else:
                old_hmm_ll = new_hmm_ll

            # M-step for each state
            for k in range(self.k):
                gamma_k = responsibilities[1:, k]

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
                    # if self.compute_convergence(new_ica_ll, old_ica_ll,
                    #                             init_ica_ll, ica_tol):
                    #     print('Converged ICA')
                    #     break

                    old_ica_ll = new_ica_ll
                    ica_iter += 1

                history['ica_ll'].append(new_ica_ll)

            # Update HMM parameters
            self._update_hmm_parameters(x, responsibilities, alpha_hat, beta_hat, c_t)

            # Check HMM convergence
            new_hmm_ll = self._compute_total_likelihood(x, responsibilities, alpha_hat, beta_hat, c_t)
            # if self.compute_convergence(new_hmm_ll, old_hmm_ll,
            #                             init_hmm_ll, hmm_tol):
            #     print(f'Converged HMM')
            #     break

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
        return tf.cast(tf.reduce_sum(gamma_k * log_probs), tf.float32)

    def _update_parameters(self, state: int, W_grad: tf.Tensor,
                           R_grad: tf.Tensor, beta_grad: tf.Tensor,
                           C_grad: Optional[tf.Tensor], ica_iter: int):
        """
        -------------------------------------------------------
        Updates model parameters using optimizers created on the fly
        multiply -1 to gradients so it maximizes the functions
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
        W_optimizer.apply_gradients([(-1*W_grad, self.ica.W[state])])

        # Update source distribution parameters
        R_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rates['R'])
        R_optimizer.apply_gradients([(-1*R_grad, self.ica.ge[state].R)])

        if self.grad_computer.use_analytical:
            self.ica.ge[state].beta = beta_grad
        else:
            beta_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rates['beta'])
            beta_optimizer.apply_gradients([(-1*beta_grad, self.ica.ge[state].beta)])

        # Update GAR coefficients if using GAR
        if self.use_gar and C_grad is not None and (ica_iter + 1) % self.update_interval == 0:
            C_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rates['C'])
            C_optimizer.apply_gradients([(-1*C_grad, self.ica.gar.C[state])])

    def _update_hmm_parameters(self, x: tf.Tensor, responsibilities: tf.Tensor, alpha_hat: tf.Tensor,
                               beta_hat: tf.Tensor, c_t: tf.Tensor):
        """
        -------------------------------------------------------
        Updates HMM transition and initial probability parameters
        -------------------------------------------------------
        Parameters:
            x - Input data (tf.Tensor of shape [time_steps, x_dims])
            responsibilities - State responsibilities (tf.tensor array of shape
                             [time_steps+1, k])
            alpha_hat - Forward pass (tf.Tensor of shape [time_steps, k])
            beta_hat - Backward pass (tf.Tensor of shape [time_steps, k])
            c_t - Normalization constants (tf.Tensor of shape [time_steps])
        -------------------------------------------------------
        """
        # Update initial state probabilities
        self.hmm.pi = responsibilities[0]

        # Update transition probabilities
        A = tf.reduce_sum(self._compute_joint_state_prob(x, alpha_hat, beta_hat, c_t), axis=0)

        # Normalize
        row_sums = tf.reduce_sum(A, axis=1)
        # row_sums[row_sums == 0] = 1  # Avoid division by zero
        row_sums = tf.where(tf.equal(row_sums, 0), tf.ones_like(row_sums), row_sums)
        self.hmm.transition_matrix = A / row_sums

    def _compute_total_likelihood(self, x: tf.Tensor, responsibilities: tf.Tensor, alpha_hat: tf.Tensor,
                                  beta_hat: tf.Tensor, c_t: tf.Tensor) -> float:
        """
        -------------------------------------------------------
        Computes complete joint log-likelihood of the HMICA model:
        L = ∑_(k=1)^K γ_k(0) log π_k +
            ∑_(t=1)^T ∑_(k'=1)^K ∑_(k=1)^K p(z_(t,k'),z_(t-1,k)|x_(1:T)) log A_(k,k') +
            ∑_(t=1)^T ∑_(k=1)^K γ_k(t) log p(x_t|z_t=k)
        -------------------------------------------------------
        Parameters:
            x - Input data (tf.Tensor of shape [time_steps, x_dims])
            responsibilities - State responsibilities (tf.Tensor of shape [time_steps+1, K])
            alpha_hat - Forward pass (tf.Tensor of shape [time_steps, K])
            beta_hat - Backward pass (tf.Tensor of shape [time_steps, K])
            c_t - Normalization constants (tf.Tensor of shape [time_steps])
        Returns:
            ll - Complete joint log-likelihood value (float)
        -------------------------------------------------------
        """
        T = x.shape[0]

        # 1. Initial state probabilities: ∑_(k=1)^K γ_k(0) log π_k
        initial_ll = tf.reduce_sum(
            responsibilities[0] * tf.math.log(self.hmm.pi)
        )

        # 2. Transition probabilities:
        # ∑_(t=1)^T ∑_(k'=1)^K ∑_(k=1)^K p(z_(t,k'),z_(t-1,k)|x_(1:T)) log A_(k,k')
        joint_prob = self._compute_joint_state_prob(x, alpha_hat, beta_hat, c_t) * tf.math.log(
            self.hmm.transition_matrix)
        transition_ll = tf.reduce_sum(joint_prob)

        # 3. ICA emission probabilities: ∑_(t=1)^T ∑_(k=1)^K γ_k(t) log p(x_t|z_t=k)
        emission_ll = 0.0
        for k in range(self.k):
            # Get log-likelihood for current state
            state_ll = self.ica.compute_log_likelihood(x, k)
            # Weight by state responsibilities
            emission_ll += tf.reduce_sum(responsibilities[1:, k] * state_ll)

        return tf.cast(initial_ll + transition_ll + emission_ll, tf.float32)

    def _compute_joint_state_prob(self, x: tf.Tensor, alpha_hat: tf.Tensor, beta_hat: tf.Tensor,
                                  c_t: tf.Tensor) -> tf.Tensor:
        """
        -------------------------------------------------------
        Computes joint probability p(z_t,z_{t-1}|x_{1:T}) using forward-backward
        variables and transition matrix
        -------------------------------------------------------
        Parameters:
            x - Input data (tf.Tensor of shape [time_steps, x_dims])
            alpha_hat - Forward pass (tf.Tensor of shape [time_steps, K])
            beta_hat - Backward pass (tf.Tensor of shape [time_steps, K])
            c_t - Normalization constants (tf.Tensor of shape [time_steps])
        Returns:
            joint_prob - Joint state probabilities (tf.Tensor of shape [K, K])
        -------------------------------------------------------
        """
        # Get forward-backward variables
        obs_prob = lambda state, x: self.ica.compute_likelihood(x, state)

        joint_prob = self.hmm.p_zt_zt1_xt(x, alpha_hat, beta_hat, c_t, obs_prob)

        return joint_prob
