"""
-------------------------------------------------------
Independent Component Analysis (ICA) implementation with optional
Generalized Autoregressive modeling
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "11/27/24"
-------------------------------------------------------
"""
from typing import Optional

# Imports
import tensorflow as tf
from generalizedAR import GeneralizedAutoRegressive

# Constants
LOG_EPSILON = 1e-8
RANDOM_SEED=1234
tf.random.set_seed(RANDOM_SEED)

class GeneralizedExponential:
    """
    -------------------------------------------------------
    A class implementing the Generalized Exponential distribution
    with shape parameter R and scale parameter β
    G(a;R,β) = (Rβ^(1/R))/(2Γ(1/R)) * exp(-β|a|^R)
    -------------------------------------------------------
    Parameters:
        m - Number of independent components (int > 0)
    -------------------------------------------------------
    """

    def __init__(self, m: int):
        self.m = m
        self.R = tf.Variable(
            tf.ones(m, dtype=tf.float32) * 2.0,  # Initialize as Gaussian; Suggested in paper (pg 16)
            name='ge_shape'
        )
        # beta_init = tf.exp(tf.math.lgamma(1/self.R)) / (2 * tf.exp(tf.math.lgamma(3/self.R)))
        beta_init = tf.ones(m, dtype=tf.float32) # Suggested in paper (pg 16)
        self.beta = tf.Variable(
            beta_init,
            name='ge_scale'
        )

    def pdf(self, a: tf.Tensor) -> tf.Tensor:
        """
        -------------------------------------------------------
        Computes the Generalized Exponential PDF
        -------------------------------------------------------
        Parameters:
            a - Input values for PDF computation (tf.Tensor of shape [..., m])
        Returns:
            pdf_values - Probability density values (tf.Tensor of same shape as input)
        -------------------------------------------------------
        """
        Z = (self.R * tf.pow(self.beta, 1 / self.R)) / (2 * tf.exp(tf.math.lgamma(1 / self.R)))
        exp_term = tf.exp(-self.beta * tf.pow(tf.abs(a) + LOG_EPSILON, self.R))
        return Z * exp_term

    def log_pdf(self, a: tf.Tensor) -> tf.Tensor:
        """
        -------------------------------------------------------
        Computes the log of the Generalized Exponential PDF
        -------------------------------------------------------
        Parameters:
            a - Input values for log PDF computation (tf.Tensor of shape [..., m])
        Returns:
            log_pdf_values - Log probability density values (tf.Tensor of same shape as input)
        -------------------------------------------------------
        """
        log_Z = (tf.math.log(self.R) + (1 / self.R) * tf.math.log(self.beta) -
                 tf.math.log(2.0) - tf.math.lgamma(1 / self.R))
        exo_term_log = -self.beta * tf.pow(tf.abs(a) + LOG_EPSILON, self.R)
        return log_Z + exo_term_log


class ICA:
    """
    -------------------------------------------------------
    Independent Component Analysis implementation supporting multiple states
    and optional Generalized Autoregressive modeling of sources
    -------------------------------------------------------
    Parameters:
        k - Number of states (int > 0)
        m - Number of sources/dimensions (int > 0)
        x_dims - Number of observed dimensions (int > 0), if None will be inferred from data
        use_gar - Whether to use Generalized Autoregressive modeling (bool)
        gar_order - Order of GAR model if used (int > 0, default=1)
    -------------------------------------------------------
    """

    def __init__(self, k: int, m: int, x_dims: Optional[int] = None, use_gar: bool = False, gar_order: int = 1):

        assert k > 0, "Number of states must be positive"
        assert m > 0, "Number of sources must be positive"
        if x_dims is not None:
            assert x_dims > 0, "Number of dimensions must be positive"

        self.k = k  # Number of states
        self.m = m  # Number of sources/dimensions
        self.use_gar = use_gar
        self.x_dim = x_dims
        self.W = None

        # Initialize a GeneralizedExponential object for each state
        self.ge = [GeneralizedExponential(m) for _ in range(k)]
        self.gar = GeneralizedAutoRegressive(k, m, gar_order) if use_gar else None

    def initialize_unmixing_matrices(self, x_dims: int) -> None:
        """
        -------------------------------------------------------
        Initializes unmixing matrices once input dimensions are known
        -------------------------------------------------------
        Parameters:
            x_dims - Number of observed dimensions (int > 0)
        -------------------------------------------------------
        """
        # if self.W is not None and self.x_dim == x_dims:
        #     return  # Already initialized with correct dimensions
        #
        # self.x_dim = x_dims
        #
        # # Generate all random matrices at once [k, x_dim, m]
        # W_init = tf.random.normal(shape=(self.k, self.x_dim, self.m))
        #
        # # QR decomposition on batched matrices
        # q, r = tf.linalg.qr(W_init)
        #
        # # Scale by Xavier/Glorot factor
        # limit = tf.math.sqrt(6 / (self.x_dim + self.m))
        # W_init = q * limit
        #
        # # Row normalize across last dimension
        # W_init = W_init / tf.norm(W_init, axis=2, keepdims=True)
        #
        # # Create list of Variables
        # self.W = [tf.Variable(W_init[i], dtype=tf.float32,
        #                       name=f'unmixing_matrices_{i}') for i in range(self.k)]
        self.W = [
            tf.Variable(
                tf.random.uniform(shape=(self.x_dim, self.m), minval=-0.05, maxval=0.05, seed=i)+tf.eye(self.x_dim, self.m),
                dtype=tf.float32,
                name=f'unmixing_matrices_{i}') for i in range(self.k)
        ]  # Suggested in paper (pg 16)


    @tf.function
    def get_sources(self, x: tf.Tensor, state: int) -> tf.Tensor:
        """
        -------------------------------------------------------
        Unmixes the input signals using the unmixing matrix for the given state
        -------------------------------------------------------
        Parameters:
            x - Input mixed signals (tf.Tensor of shape [time_steps, x_dim])
            state - Current state index (int, 0 <= state < k)
        Returns:
            sources - Unmixed source signals (tf.Tensor of shape [time_steps, m])
        -------------------------------------------------------
        """
        assert 0 <= state < self.k, "Invalid state index"

        # Initialize unmixing matrices if needed
        if self.W is None:
            self.initialize_unmixing_matrices(x.shape[-1])
        return tf.matmul(x, self.W[state], transpose_b=False)

    @tf.function
    def _compute_log_det(self, state: int) -> tf.Tensor:
        """
        -------------------------------------------------------
        Computes the log pseudo determinant of the unmixing matrix for the given state
        Would re-use computation with computation graphs
        -------------------------------------------------------
        Parameters:
            state - Current state index (int, 0 <= state < k)
        Returns:
            log_det - Log determinant of the unmixing matrix (tf.Tensor of shape [])
        -------------------------------------------------------
        """

        # matrix is well-conditioned when computing pseudo determinant; prevents Eigen::BDCSVD failed with error code
        W_scaled = self.W[state] / (tf.norm(self.W[state], axis=1, keepdims=True) + LOG_EPSILON)

        # Add small regularization to support pseudo-determinant stability
        reg_matrix = W_scaled + tf.eye(W_scaled.shape[0], W_scaled.shape[1]) * 1e-6

        try:
            s = tf.linalg.svd(reg_matrix, compute_uv=False)
        except:
            # Compute QR decomposition first
            q, r = tf.linalg.qr(reg_matrix)
            # Get singular values from the R matrix (more stable)
            s = tf.abs(tf.linalg.diag_part(r))

        return tf.reduce_sum(tf.math.log(tf.abs(s)))

    @tf.function
    def compute_likelihood(self, x: tf.Tensor, state: int) -> tf.Tensor:
        """
        -------------------------------------------------------
        Computes the likelihood of the data given the state:
        p(x_t|z_k) = p(a_t)/|J| where |J| = 1/det(W)
        p(x_t|z_k) = ∏_i p(a_i[t]) * |det(W)|
        -------------------------------------------------------
        Parameters:
            x - Input mixed signals (tf.Tensor of shape [time_steps, x_dim])
            state - Current state index (int, 0 <= state < k)
        Returns:
            likelihood - Likelihood values for each time step (tf.Tensor of shape [time_steps] | [time_steps-gar_order])
        -------------------------------------------------------
        """
        assert 0 <= state < self.k, "Invalid state index"

        # Initialize unmixing matrices if needed
        if self.W is None:
            self.initialize_unmixing_matrices(x.shape[-1])

        # Get unmixed sources
        sources = self.get_sources(x, state)

        # Compute pseudo-determinant term for non-square matrix
        log_det = self._compute_log_det(state)

        # Compute source probabilities
        if self.use_gar:
            # Use prediction errors from GAR model
            errors = self.gar.compute_prediction_error(sources, state)
            # We lose p time steps due to GAR prediction
            sources = sources[self.gar.p:]
            source_probs = self.ge[state].pdf(errors)
        else:
            source_probs = self.ge[state].pdf(sources)

        # Multiply probabilities across sources
        total_probs = tf.reduce_prod(source_probs, axis=1)

        # Apply Jacobian transformation
        likelihood = total_probs * tf.exp(log_det)

        return likelihood

    @tf.function
    def compute_log_likelihood(self, x: tf.Tensor, state: int) -> tf.Tensor:
        """
        -------------------------------------------------------
        Computes the log-likelihood of the data given the state:
        log p(x_t|z_k) = log|det(W)| + sum_i log p(a_i[t])
        -------------------------------------------------------
        Parameters:
            x - Input mixed signals (tf.Tensor of shape [time_steps, x_dim])
            state - Current state index (int, 0 <= state < k)
        Returns:
            log_likelihood - Log-likelihood values for each time step
                           (tf.Tensor of shape [time_steps] | [time_steps-gar_order])
        -------------------------------------------------------
        """
        assert 0 <= state < self.k, "Invalid state index"

        # Initialize unmixing matrices if needed
        if self.W is None:
            self.initialize_unmixing_matrices(x.shape[-1])

        # Get unmixed sources
        sources = self.get_sources(x, state)

        # Compute log determinant term
        log_det = self._compute_log_det(state)

        # Compute source log probabilities
        if self.use_gar:
            # Use prediction errors from GAR model
            errors = self.gar.compute_prediction_error(sources, state)
            # We lose p time steps due to GAR prediction
            sources = sources[self.gar.p:]
            source_log_probs = self.ge[state].log_pdf(errors)
        else:
            source_log_probs = self.ge[state].log_pdf(sources)

        # Sum log probabilities across sources
        total_log_probs = tf.reduce_sum(source_log_probs, axis=1)

        # Add log determinant term
        log_likelihood = total_log_probs + log_det

        return log_likelihood
