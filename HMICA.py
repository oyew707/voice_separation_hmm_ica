"""
-------------------------------------------------------
Sklearn-style wrapper for Hidden Markov Independent Component Analysis
-------------------------------------------------------
Author:  einsteinoyewole
Email:   [your email address]
__updated__ = "12/9/24"
-------------------------------------------------------
"""


# Imports
from typing import Optional, Dict
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted
from hmica_learning import HMICALearner, ParallelHMICALearner


# Constants


class HMICA_SKLEARN(BaseEstimator, TransformerMixin):
    """
    -------------------------------------------------------
    Sklearn-style wrapper for Hidden Markov Independent Component Analysis
    -------------------------------------------------------
    Parameters:
        n_sources - Number of independent sources to estimate (int > 0)
        n_states - Number of states in the Hidden Markov Model (int > 0)
        whiten - if True, performs whitening of input data before fitting (bool)
        use_gar - Use Generalized AutoRegressive model for source extraction (bool)
        gar_order - Order of the Generalized AutoRegressive model if use_gar (int > 0)
        inference_method - Inference method for HMICA (str; 'viterbi' or 'smoothing')
        initial_params - Initial parameters for HMICA model (Dict)
                        - 'W' - Unmixing matrix
                        - 'R' - Shape parameter of generalized exponential density
                        - 'beta' - Scale parameter of generalized exponential density
                        - 'C' - AR coefficients for GAR model
                        - 'pi' - Initial state probabilities
                        - 'A' - Transition matrix
        learning_rate - Learning rate in HMICA model for different parameters (Dict[str, float])
                        - 'W' - Unmixing matrix
                        - 'R' - Shape parameter of generalized exponential density
                        - 'beta' - Scale parameter of generalized exponential density
                        - 'C' - AR coefficients for GAR model
        update_interval - Number of ICA iterations before updating GAR parameters (int > 0)
        max_iter - Maximum number of iterations for HMM and ICA optimization. (Dict[str, int])
                        - 'hmm' - Maximum number of iterations for HMM optimization
                        - 'ica' - Maximum number of iterations for ICA optimization
        tol - Tolerance for convergence in HMICA model (Dict[str, float])
                        - 'hmm' - Tolerance for HMM optimization
                        - 'ica' - Tolerance for ICA optimization
        optimizer_patience - Number of iterations with no improvement before early stopping. (str; 'adam' or 'sgd')
        warmup_period - Number of intial iterations before checking for convergence (int > 0)
        random_seed - Seed for random initialization (Optional[int])
        n_jobs - Number of parallel jobs to run for HMICA (int > 0) if None use simple learner
    -------------------------------------------------------
    """

    def __init__(
            self,
            n_states: int = 2,
            n_sources: Optional[int] = None,
            whiten: bool = False,
            use_gar: bool = True,
            gar_order: int = 2,
            inference_method: str = 'viterbi',
            learning_rates: Optional[Dict[str, float]] = None,
            update_interval: int = 10,
            initial_params: Optional[Dict] = None,
            max_iter: Dict[str, int] = dict(hmm=10, ica=1000),
            tol: Dict[str, float] = dict(hmm=1e-4, ica=1e-2),
            optimizer_patience: int = 5,
            warmup_period: int = 5,
            n_processes: Optional[int] = None,
            random_state: Optional[int] = None
    ):
        self.n_states = n_states
        self.n_sources = n_sources
        self.whiten = whiten
        self.use_gar = use_gar
        self.gar_order = gar_order
        self.inference_method = inference_method
        self.learning_rates = learning_rates
        self.update_interval = update_interval
        self.initial_params = initial_params or {}
        self.max_iter = max_iter
        self.tol = tol
        self.optimizer_patience = optimizer_patience
        self.warmup_period = warmup_period
        self.n_processes = n_processes
        self.random_state = random_state

        # Internal attributes
        self.scaler_ = PCA(whiten=True, n_components=None) if whiten else None

    def _validate_params(self):
        """
        -------------------------------------------------------
        Validates parameters for HMICA model.
        -------------------------------------------------------
        """
        if self.inference_method not in ['viterbi', 'smoothing']:
            raise ValueError("inference_method must be either 'viterbi' or 'smoothing'")

        if self.n_sources is not None and self.n_sources > self.n_features_in_:
            raise ValueError("n_sources cannot be greater than n_features")

        if self.use_gar and self.gar_order < 1:
            raise ValueError("gar_order must be >= 1 when use_gar is True")

    def _initialize_model(self, X):
        """
        -------------------------------------------------------
        Initializes the HMICA model with the specified parameters.
        -------------------------------------------------------
        Parameters:
           X - array-like of shape (n_samples, n_features)
        -------------------------------------------------------
        """
        # Set number of sources if not specified
        if self.n_sources is None:
            self.n_sources_ = X.shape[1]
        else:
            self.n_sources_ = self.n_sources

        # Set random seed if specified
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
            np.random.seed(self.random_state)

        # Create model class
        model_class = ParallelHMICALearner if self.n_processes else HMICALearner

        # Initialize model
        self.model_ = model_class(
            k=self.n_states,
            m=self.n_sources_,
            x_dims=X.shape[1],
            use_gar=self.use_gar,
            gar_order=self.gar_order,
            update_interval=self.update_interval,
            learning_rates=self.learning_rates,
            use_analytical=False,
            A=self.initial_params.get('A'),
            pi=self.initial_params.get('pi'),
            random_seed=self.random_state,
            n_workers=self.n_processes,
            patience=self.optimizer_patience,
            warmup_period=self.warmup_period
        )

        # Initialize model parameters if provided
        self.model_.set_ica_parameters(W=self.initial_params.get('W'), R=self.initial_params.get('R'),
                                       beta=self.initial_params.get('beta'), C=self.initial_params.get('C'))

    def fit(self, X, y=None):
        """
        -------------------------------------------------------
        Fit the HMICA model to the data.
        -------------------------------------------------------
        Parameters:
           X - array-like of shape (n_samples, n_features)
           y - Ignored, present for compatibility
        Returns:
           self - Fitted HMICA model instance
        -------------------------------------------------------
        """
        # Convert input to tensor if needed
        if not isinstance(X, tf.Tensor):
            X = tf.convert_to_tensor(X, dtype=tf.float32)

        # Store number of features -for sklearn estimator
        self.n_features_in_ = X.shape[1]
        # Validate parameters
        self._validate_params()

        # Initialize model
        self._initialize_model(X)

        # Apply whitening if requested
        if self.whiten:
            X = self.scaler_.fit_transform(X)
            X = tf.convert_to_tensor(X, dtype=tf.float32)

        # Train model
        self.training_history_ = self.model_.train(
            x=X,
            hmm_max_iter=self.max_iter['hmm'],
            ica_max_iter=self.max_iter['ica'],
            hmm_tol=self.tol['hmm'],
            ica_tol=self.tol['ica']
        )
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        -------------------------------------------------------
        Fit the HMICA model to the data to learn state, unmixing matrix and source signals.
        -------------------------------------------------------
        Parameters:
           X - array-like of shape (n_samples, n_features)
           y - Ignored, present for compatibility
        Returns:
          source_signals - Estimated source signals (array-like of shape (n_samples, n_sources))
           state_sequence - State sequence of the HMM model (array-like of shape (n_samples, k))
        -------------------------------------------------------
        """
        return self.fit(X, **fit_params).transform(X)

    def transform(self, X):
        """
        -------------------------------------------------------
        Performs the blind source separation
        -------------------------------------------------------
        Parameters:
           X - array-like of shape (n_samples, n_features)
        Returns:
           source_signals - Estimated source signals (array-like of shape (n_samples, n_sources))
           state_sequence - State sequence of the HMM model (array-like of shape (n_samples, k))
        -------------------------------------------------------
        """
        check_is_fitted(self, ['model_'])

        # Apply whitening if used during fitting
        if self.whiten:
            X = self.scaler_.fit_transform(X)
            X = tf.convert_to_tensor(X, dtype=tf.float32)

        # Convert input to tensor if needed
        if not isinstance(X, tf.Tensor):
            X = tf.convert_to_tensor(X, dtype=tf.float32)

        # Get state sequence
        obs_prob = lambda state, x: self.model_.ica.compute_likelihood(x, state)
        if self.inference_method == 'viterbi':
            state_sequence, _ = self.model_.hmm.viterbi_for_inference(X, obs_prob)
        else:
            state_sequence = self.model_.hmm.infer_states_forward_backward(X, obs_prob)[1:]

        # Get sources for each state
        sources_k = [self.model_.ica.get_sources(X, k) for k in range(self.n_states)]
        sources_stacked = tf.stack(sources_k)
        source_signal = tf.einsum('tk,ktn->tn', state_sequence, sources_stacked)
        return {
            'source_signals': source_signal,
            'state_sequence': state_sequence
        }

    def inverse_transform(self, S, state_sequence):
        """
        -------------------------------------------------------
        Converts source signals back to the original observation.
        -------------------------------------------------------
        Parameters:
           S - Source signals array-like of shape (n_samples, n_sources)
              state_sequence - State sequence of the HMM model (array-like of shape (n_samples, k))
        Returns:
           X - mixed signals array-like of shape (n_samples, n_features)
        -------------------------------------------------------
        """
        check_is_fitted(self, ['model_'])

        # Convert input to tensor if needed
        if not isinstance(S, tf.Tensor):
            S = tf.convert_to_tensor(S, dtype=tf.float32)

        # Get mixture for each state
        mixtures_k = [self.model_.ica.get_mixture(S, k) for k in range(self.n_states)]
        mixtures_stacked = tf.stack(mixtures_k)

        X = tf.einsum('tk,ktn->tn', state_sequence, mixtures_stacked)

        # Reverse whitening if requested
        if self.whiten:
            X = self.scaler_.inverse_transform(X)
            X = tf.convert_to_tensor(X, dtype=tf.float32)

        return X

    def score(self, X, y=None):
        """
        -------------------------------------------------------
        Returns the log-likelihood of the data under the model.
        -------------------------------------------------------
        Parameters:
              X - array-like of shape (n_samples, n_features)
              y - Ignored, present for compatibility
        Returns:
                log_likelihood - Log-likelihood of the data under the model
        -------------------------------------------------------
        """
        # Convert input to tensor if needed
        if not isinstance(X, tf.Tensor):
            X = tf.convert_to_tensor(X, dtype=tf.float32)

        obs_prob = self.model_.hmm.calc_obs_prob(X, lambda state, x: self.model_.ica.compute_likelihood(x, state))
        alpha_hat, c_t = self.model_.hmm.calc_alpha_hat(X, obs_prob)
        beta_hat = self.model_.hmm.calc_beta_hat(X, obs_prob, c_t)
        g = self.model_.hmm.p_zt_xT(X, alpha_hat, beta_hat, c_t)
        responsibilities = g / tf.reduce_sum(g, axis=1, keepdims=True)
        ll = self.model_._compute_total_likelihood(X, responsibilities, alpha_hat, beta_hat, c_t)
        return float(ll)


class HMICA(HMICA_SKLEARN):
    pass
