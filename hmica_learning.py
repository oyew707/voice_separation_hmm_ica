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
from typing import Optional, Dict, List
from ICA import ICA, GeneralizedExponential
from discreteHMM import DiscreteHMM
from generalizedAR import GeneralizedAutoRegressive
from ica_gradients import ICAGradients
from tqdm import tqdm
from functools import wraps, partial
import time
from collections import defaultdict
import multiprocessing as mp

# Constants
tf.config.run_functions_eagerly(True)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


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
        learning_rates - Dictionary of learning rates for model parameters (Optional[dict])
        use_analytical - Whether to use analytical gradients for GAR (bool)
        A - Initial transition matrix for HMM (Optional[tf.Tensor])
        pi - Initial state probabilities for HMM (Optional[tf.Tensor])
    -------------------------------------------------------
    """

    def __init__(self, k: int, m: int, x_dims: int, use_gar: bool = False,
                 gar_order: int = 1, update_interval: int = 10, learning_rates: Optional[dict] = None,
                 use_analytical: bool = True, A: Optional[tf.Tensor] = None, pi: Optional[tf.Tensor] = None,
                 random_seed: Optional[int] = None, **kwargs):
        # Initialize models
        self.k = k
        self.m = m
        self.x_dims = x_dims
        self.use_gar = use_gar
        self.update_interval = update_interval
        self.gar_order = gar_order
        self.random_seed = random_seed
        self.patience = kwargs['patience']
        self.warmup_steps = kwargs['warmup_period']

        # Default learning rates
        default_rates = {
            'W': 1e-4,  # Unmixing matrix
            'R': 1e-4,  # Shape parameter
            'beta': 1e-4,  # Scale parameter
            'C': 1e-4  # GAR coefficients
        }
        self.learning_rates = learning_rates | default_rates if learning_rates is not None else default_rates

        # set seed
        tf.random.set_seed(random_seed)

        # Initialize HMM with uniform initial probabilities and transitions
        diag_prob = 0.35
        off_diag_prob = (1.0 - diag_prob) / (k - 1)

        transition_matrix = tf.ones((k, k)) * off_diag_prob
        transition_matrix += tf.eye(k) * (diag_prob - off_diag_prob)

        A = transition_matrix if A is None else A
        pi = tf.ones(k) / k if pi is None else pi
        self.hmm = DiscreteHMM(pi, A)

        # Initialize ICA model
        self.ica = ICA(k, m, x_dims, use_gar, gar_order)

        # Initialize gradient computer
        self.grad_computer = ICAGradients(use_analytical)

    def set_ica_parameters(self, W: Optional[tf.Tensor], R: Optional[tf.Tensor], beta: Optional[tf.Tensor],
                           C: Optional[tf.Tensor]):
        """
        -------------------------------------------------------
        Set initial parameters for ICA model
        -------------------------------------------------------
        Parameters:
            W - Initial unmixing matrix (tf.Tensor of shape [m, m])
            R - Initial shape parameter (tf.Tensor of shape [m])
            beta - Initial scale parameter (tf.Tensor of shape [m])
            C - Initial GAR coefficients (tf.Tensor of shape [m, p])
        -------------------------------------------------------
        """
        # Initialize ICA model with W
        self.ica = ICA(self.k, self.m, self.x_dims, self.use_gar, self.gar_order, W=W, random_seed=self.random_seed)
        # Initialize GE distributions
        self.ica.ge = [GeneralizedExponential(self.m, R=R[i] if R else None, beta=beta[i] if beta else None) for i in range(self.k)]
        if self.use_gar:
            self.ica.gar = GeneralizedAutoRegressive(self.k, self.m, self.gar_order, self.random_seed, C)

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

    def train(self, x: tf.Tensor, hmm_max_iter: int = 100, ica_max_iter: int = 10,
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
        history = {'hmm_ll': [], 'ica_ll': defaultdict(list)}
        init_hmm_ll, new_hmm_ll = None, None

        # Main EM loop
        for iteration in tqdm(range(hmm_max_iter)):
            # E-step: Forward-backward algorithm
            obs_prob = self.hmm.calc_obs_prob(x, lambda state, x: self.ica.compute_likelihood(x, state))
            alpha_hat, c_t = self.hmm.calc_alpha_hat(x, obs_prob)
            beta_hat = self.hmm.calc_beta_hat(x, obs_prob, c_t)
            g = self.hmm.p_zt_xT(x, alpha_hat, beta_hat, c_t)
            responsibilities = g / tf.reduce_sum(g, axis=1, keepdims=True)

            # Initial likelihood computation
            if init_hmm_ll is None:
                init_hmm_ll = self._compute_total_likelihood(x, responsibilities, alpha_hat, beta_hat, c_t)
                old_hmm_ll = init_hmm_ll

            # M-step for each state
            for k in range(self.k):
                gamma_k = responsibilities[1:, k]

                # Initialize ICA iteration tracking
                init_ica_ll = self._compute_ica_likelihood(x, k, gamma_k)
                old_ica_ll = init_ica_ll
                patience = 0
                best_ica_ll = init_ica_ll

                # ICA update loop
                for ica_iter in range(ica_max_iter):

                    # Compute and apply gradients
                    W_grad, R_grad, beta_grad, C_grad = self.grad_computer.compute_gradients(
                        x, gamma_k, k, self.ica)

                    # Update parameters using gradient steps
                    self._update_parameters(k, W_grad, R_grad, beta_grad, C_grad, ica_iter)

                    # Check ICA convergence
                    new_ica_ll = self._compute_ica_likelihood(x, k, gamma_k)
                    if ica_iter >= self.warmup_steps and self.compute_convergence(new_ica_ll, old_ica_ll, init_ica_ll,
                                                                                  ica_tol):
                        # print(f'Converged ICA {k} {ica_iter} in {iteration + 1} iterations')
                        break
                    early_stop, patience, best_ica_ll = self._early_stopping(best_ica_ll, new_ica_ll,
                                                                             ica_iter, patience)
                    if early_stop:
                        # print(f'Early stop ICA {k} {ica_iter} in {iteration + 1} iterations')
                        break

                    old_ica_ll = new_ica_ll

                    history['ica_ll'][k].append(float(new_ica_ll))

            # Update HMM parameters
            self._update_hmm_parameters(x, responsibilities, alpha_hat, beta_hat, c_t)

            # Check HMM convergence
            new_hmm_ll = self._compute_total_likelihood(x, responsibilities, alpha_hat, beta_hat, c_t)
            if iteration > self.warmup_steps and self.compute_convergence(new_hmm_ll, old_hmm_ll, init_hmm_ll, hmm_tol):
                # print(f'Converged HMM in {iteration + 1} iterations')
                break

            old_hmm_ll = new_hmm_ll
            history['hmm_ll'].append(float(new_hmm_ll))

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
        W_optimizer.apply_gradients([(-1 * W_grad, self.ica.W[state])])

        if (ica_iter + 1) % self.update_interval != 0:
            return

        # Update source distribution parameters
        R_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rates['R'])
        R_optimizer.apply_gradients([(-1 * R_grad, self.ica.ge[state].R)])

        if self.grad_computer.use_analytical:
            self.ica.ge[state].beta = beta_grad
        else:
            beta_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rates['beta'])
            beta_optimizer.apply_gradients([(-1 * beta_grad, self.ica.ge[state].beta)])

        # Update GAR coefficients if using GAR
        if self.use_gar and C_grad is not None:
            C_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rates['C'])
            C_optimizer.apply_gradients([(-1 * C_grad, self.ica.gar.C[state])])

    @timeit
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
        row_sums = tf.reduce_sum(A, axis=1, keepdims=True)
        # row_sums[row_sums == 0] = 1  # Avoid division by zero
        row_sums = tf.where(tf.equal(row_sums, 0), tf.ones_like(row_sums), row_sums)
        self.hmm.transition_matrix = A / row_sums

        # Add small epsilon to avoid zeros
        epsilon = 1e-6
        self.hmm.transition_matrix += epsilon
        # Renormalize after adding epsilon
        self.hmm.transition_matrix /= tf.reduce_sum(self.hmm.transition_matrix, axis=1, keepdims=True)

    @timeit
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

    def _early_stopping(self, best_loss, current_loss, current_iteration, patience):
        """
        -------------------------------------------------------
        Early stopping criterion based on loss values
        -------------------------------------------------------
        Parameters:
            best_loss - Best loss value so far (float)
            current_loss - Current loss value (float)
            current_iteration - Current iteration number (int)
            patience - Current patience value (int)
        Returns:
            stop - Whether to stop training (bool)
            patience - Updated patience value (int)
            best_loss - Updated bestloss value (float)
        -------------------------------------------------------
        """
        if current_iteration < self.warmup_steps:
            return False, patience + 1, best_loss
        if current_loss > best_loss:
            best_loss = current_loss
            patience = 0
        else:
            patience += 1
        return patience >= self.patience, patience, best_loss


class ParallelHMICALearner(HMICALearner):
    """
    -------------------------------------------------------
    Parallel M-step implementation for Hidden Markov Independent
    Component Analysis using multiprocessing
    -------------------------------------------------------
    Parameters:
        n_workers - Number of parallel workers (int > 0)
    """

    def __init__(self, *args, n_workers: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_workers = n_workers or mp.cpu_count()
        self.ica_max_iter = None
        self.ica_tol = None

    def _parallel_state_optimization(self, x: tf.Tensor, responsibilities: tf.Tensor,
                                     state: int) -> Dict[str, tf.Tensor]:
        """
        -------------------------------------------------------
        Optimizes parameters for a single state in a separate process
        -------------------------------------------------------
        Parameters:
            x - Input mixed signals (tf.Tensor of shape [time_steps, x_dim])
            responsibilities - State responsibilities (tf.Tensor of shape [time_steps+1, k])
            state - Current state index to optimize (int, 0 <= state < k)
        Returns:
            results - Dictionary containing optimized parameters:
                     'W': Unmixing matrix for the state
                     'R': Shape parameters for GE distribution
                     'beta': Scale parameters for GE distribution
                     'C': GAR coefficients (if using GAR)
                     'final_ll': Final log-likelihood value
        -------------------------------------------------------
        """
        assert 0 <= state < self.k, "Invalid state index"
        assert self.ica_max_iter is not None, "ICA max iterations not set"
        assert self.ica_tol is not None, "ICA convergence tolerance not set"

        gamma_k = responsibilities[1:, state]

        # Initialize ICA iteration tracking
        init_ica_ll = self._compute_ica_likelihood(x, state, gamma_k)
        new_ica_ll = old_ica_ll = init_ica_ll
        patience = 0
        best_ica_ll = init_ica_ll

        # ICA update loop
        for ica_iter in range(self.ica_max_iter):
            # Compute and apply gradients
            W_grad, R_grad, beta_grad, C_grad = self.grad_computer.compute_gradients(
                x, gamma_k, state, self.ica)

            # Update parameters using gradient steps
            self._update_parameters(state, W_grad, R_grad, beta_grad, C_grad, ica_iter)

            # Check ICA convergence
            new_ica_ll = self._compute_ica_likelihood(x, state, gamma_k)
            if ica_iter >= self.warmup_steps and self.compute_convergence(new_ica_ll, old_ica_ll, init_ica_ll,
                                                                          self.ica_tol):
                # print(f'Converged ICA {state} in {ica_iter} iterations')
                break
            early_stop, patience, best_ica_ll = self._early_stopping(best_ica_ll, new_ica_ll, patience, self.ica_tol)
            if ica_iter >= self.warmup_steps and early_stop:
                break

            old_ica_ll = new_ica_ll

        return {
            'W': self.ica.W[state].numpy(),
            'R': self.ica.ge[state].R.numpy(),
            'beta': self.ica.ge[state].beta.numpy(),
            'C': self.ica.gar.C[state].numpy() if self.ica.use_gar else None,
            'final_ll': float(new_ica_ll)
        }

    def train(self, x: tf.Tensor, hmm_max_iter: int = 100, ica_max_iter: int = 10,
              hmm_tol: float = 1e-4, ica_tol: float = 1e-2) -> dict:
        """
        -------------------------------------------------------
        Trains the HMICA model using parallel M-step optimization
        -------------------------------------------------------
        Parameters:
            x - Input mixed signals (tf.Tensor of shape [time_steps, x_dim])
            hmm_max_iter - Maximum number of HMM EM iterations (int > 0)
            ica_max_iter - Maximum number of ICA iterations per state (int > 0)
            hmm_tol - Convergence tolerance for HMM updates (float > 0)
            ica_tol - Convergence tolerance for ICA updates (float > 0)
        Returns:
            history - Dictionary containing:
                     'hmm_ll': List of HMM log-likelihood values
                     'ica_ll': List of ICA log-likelihood values
        -------------------------------------------------------
        """
        self.ica_max_iter = ica_max_iter
        self.ica_tol = ica_tol
        history = {'hmm_ll': [], 'ica_ll': defaultdict(list)}
        init_hmm_ll, new_hmm_ll = None, None

        # Create process pool
        pool = mp.Pool(processes=self.n_workers)

        # Main EM loop
        for iteration in tqdm(range(hmm_max_iter)):
            # E-step: Forward-backward algorithm
            obs_prob = self.hmm.calc_obs_prob(x, lambda state, x: self.ica.compute_likelihood(x, state))
            alpha_hat, c_t = self.hmm.calc_alpha_hat(x, obs_prob)
            beta_hat = self.hmm.calc_beta_hat(x, obs_prob, c_t)
            g = self.hmm.p_zt_xT(x, alpha_hat, beta_hat, c_t)
            responsibilities = g / tf.reduce_sum(g, axis=1, keepdims=True)

            # Initial likelihood computation
            if init_hmm_ll is None:
                init_hmm_ll = self._compute_total_likelihood(x, responsibilities, alpha_hat, beta_hat, c_t)
                old_hmm_ll = init_hmm_ll

            # M-step: Parallel optimization for each state
            optimize_state = partial(self._parallel_state_optimization, x, responsibilities)
            state_results = pool.map(optimize_state, range(self.k))

            # Update model parameters from parallel results
            for state, result in enumerate(state_results):
                self.ica.W[state].assign(result['W'])
                self.ica.ge[state].R.assign(result['R'])
                self.ica.ge[state].beta.assign(result['beta'])
                if self.ica.use_gar and result['C'] is not None:
                    self.ica.gar.C[state].assign(result['C'])
                history['ica_ll'][state].append(result['final_ll'])

            # Update HMM parameters
            self._update_hmm_parameters(x, responsibilities, alpha_hat, beta_hat, c_t)

            # Check HMM convergence
            new_hmm_ll = self._compute_total_likelihood(x, responsibilities, alpha_hat, beta_hat, c_t)
            if iteration > self.warmup_steps and self.compute_convergence(new_hmm_ll, old_hmm_ll, init_hmm_ll, hmm_tol):
                # print(f'Converged HMM in {iteration + 1} iterations')
                break

            old_hmm_ll = new_hmm_ll
            history['hmm_ll'].append(float(new_hmm_ll))

        pool.close()
        pool.join()
        return history
