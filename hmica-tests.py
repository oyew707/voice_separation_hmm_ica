"""
-------------------------------------------------------
Test suite for Hidden Markov Independent Component Analysis (HMICA)
Includes synthetic data generation and test cases
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "11/28/24"
-------------------------------------------------------
"""

# Imports
import tensorflow as tf
import numpy as np
from typing import Tuple, List
import unittest
from hmica_learning import HMICALearner
from scipy import signal
from evaluation import signal_to_noise_ratio, signal_to_distortion_ratio
from HMICA import HMICA
from sklearn.utils.estimator_checks import check_estimator
from sklearn.exceptions import NotFittedError


class TestDataGenerator:
    """
    -------------------------------------------------------
    Generates synthetic test data for HMICA algorithm testing
    -------------------------------------------------------
    """

    @staticmethod
    def generate_multisin_sources(n_samples: int, frequencies: List[float],
                                  noise_type: str = 'laplacian',
                                  snr: float = 10.0) -> tf.Tensor:
        """
        -------------------------------------------------------
        Generates multiple sinusoidal sources with specified noise
        -------------------------------------------------------
        Parameters:
            n_samples - Number of time points (int > 0)
            frequencies - List of frequencies for sinusoids (List[float])
            noise_type - Type of noise ('gaussian' or 'laplacian')
            snr - Signal to noise ratio (float > 0)
        Returns:
            sources - Generated source signals (tf.Tensor of shape [n_samples, len(frequencies)])
        -------------------------------------------------------
        """
        t = np.linspace(0, 10, n_samples)
        sources = np.zeros((n_samples, len(frequencies)))

        # Generate sinusoidal signals
        for i, freq in enumerate(frequencies):
            sources[:, i] = np.sin(2 * np.pi * freq * t)

        # Add noise
        signal_power = np.mean(sources ** 2)
        noise_power = signal_power / (10 ** (snr / 10))

        if noise_type == 'gaussian':
            noise = np.random.normal(0, np.sqrt(noise_power), sources.shape)
        else:  # laplacian
            scale = np.sqrt(noise_power / 2)
            noise = np.random.laplace(0, scale, sources.shape)

        return tf.convert_to_tensor(sources + noise, dtype=tf.float32)

    @staticmethod
    def mix_sources(sources: tf.Tensor, mixing_matrix: tf.Tensor) -> tf.Tensor:
        """
        -------------------------------------------------------
        Mixes source signals using provided mixing matrix
        -------------------------------------------------------
        Parameters:
            sources - Source signals (tf.Tensor of shape [n_samples, n_sources])
            mixing_matrix - Mixing matrix (tf.Tensor of shape [n_obs, n_sources])
        Returns:
            mixed - Mixed signals (tf.Tensor of shape [n_samples, n_obs])
        -------------------------------------------------------
        """
        return tf.matmul(sources, mixing_matrix, transpose_b=True)

    @staticmethod
    def generate_switching_data(n_samples: int, n_states: int = 2) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        -------------------------------------------------------
        Generates data that switches between different mixing matrices
        -------------------------------------------------------
        Parameters:
            n_samples - Number of time points per state (int > 0)
            n_states - Number of different states/mixing matrices (int > 0)
        Returns:
            mixed_signals - Mixed signals (tf.Tensor)
            true_states - True state sequence (tf.Tensor)
        -------------------------------------------------------
        """
        # Generate different mixing matrices for each state
        mixing_matrices = []
        for _ in range(n_states):
            theta = np.random.uniform(0, 2 * np.pi)
            mixing_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                      [np.sin(theta), np.cos(theta)]])
            mixing_matrices.append(mixing_matrix)

        # Generate sources and mix them differently for each state
        total_samples = n_samples * n_states
        sources = TestDataGenerator.generate_multisin_sources(
            total_samples,
            frequencies=[10, 20]
        )

        mixed_signals = tf.zeros((total_samples, 2))
        true_states = tf.zeros(total_samples, dtype=tf.int32)

        for i in range(n_states):
            start_idx = i * n_samples
            end_idx = (i + 1) * n_samples

            state_mixed = TestDataGenerator.mix_sources(
                sources[start_idx:end_idx],
                tf.convert_to_tensor(mixing_matrices[i], dtype=tf.float32)
            )
            mixed_signals = tf.tensor_scatter_nd_update(
                mixed_signals,
                tf.reshape(tf.range(start_idx, end_idx), [-1, 1]),
                state_mixed
            )
            true_states = tf.tensor_scatter_nd_update(
                true_states,
                tf.reshape(tf.range(start_idx, end_idx), [-1, 1]),
                tf.ones(n_samples, dtype=tf.int32) * i
            )

        return mixed_signals, true_states


class TestHMICA(unittest.TestCase):
    def setUp(self):
        """Set up test data and model."""
        self.n_samples = 50
        self.n_states = 2
        self.n_sources = 2
        self.x_dims = 2

        # Generate test data
        self.mixed_signals, self.true_states = TestDataGenerator.generate_switching_data(
            self.n_samples,
            self.n_states
        )

        # Initialize model
        self.model = HMICALearner(
            k=self.n_states,
            m=self.n_sources,
            x_dims=self.x_dims,
            use_gar=True,
            gar_order=4,
            use_analytical=False,
            learning_rates={'W': 1e-3, 'R': 1e-3, 'beta': 1e-3, 'C': 1e-3}
        )

    def test_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.k, self.n_states)
        self.assertEqual(self.model.m, self.n_sources)
        self.assertEqual(self.model.x_dims, self.x_dims)

    def test_training(self):
        """Test model training"""
        history = self.model.train(self.mixed_signals, hmm_max_iter=10, ica_max_iter=2)

        # Check if training history contains expected keys
        self.assertIn('hmm_ll', history)
        self.assertIn('ica_ll', history)

        print(history['hmm_ll'])
        print(history['ica_ll'])
        # Check if log likelihood improves
        self.assertGreater(history['hmm_ll'][-1], history['hmm_ll'][0])

    def test_source_separation(self):
        """Test source separation quality"""
        # Get sources for before training state
        sources_pre = self.model.ica.get_sources(
            self.mixed_signals[self.true_states == 0],
            state=0
        )

        # Train model
        self.model.train(self.mixed_signals, hmm_max_iter=10, ica_max_iter=2)

        # Get sources for first state
        sources = self.model.ica.get_sources(
            self.mixed_signals[self.true_states == 0],
            state=0
        )

        # Check source dimensions
        self.assertEqual(sources.shape[1], self.n_sources)

        # Check if sources are more independent than before training
        sp_corr = np.corrcoef(sources_pre.numpy().T)
        source_corr = np.corrcoef(sources.numpy().T)

        # Off-diagonal correlations should be smaller for sources before training
        sp_off_diag = np.abs(sp_corr[0, 1])
        source_off_diag = np.abs(source_corr[0, 1])
        self.assertGreater(sp_off_diag, source_off_diag)

    # def test_state_estimation(self):
    #     """Test state estimation accuracy"""
    #     # Train model
    #     self.model.train(self.mixed_signals, max_iter=10)
    #
    #     # Get state responsibilities
    #     obs_prob = lambda state, x: self.model.ica.compute_likelihood( x, state)
    #     obs_prob_matrix = self.model.hmm.calc_obs_prob(self.mixed_signals, obs_prob)
    #     alpha_hat, c_t = self.model.hmm.calc_alpha_hat(
    #         self.mixed_signals,
    #         obs_prob_matrix
    #     )
    #
    #     # Get most likely states
    #     estimated_states = tf.argmax(alpha_hat, axis=1).numpy().astype(np.int32)
    #
    #     # Compute accuracy (accounting for possible state permutation)
    #     accuracy = max(
    #         np.mean(estimated_states == self.true_states),
    #         np.mean(estimated_states == 1 - self.true_states)
    #     )
    #
    #     # Should be better than random guessing
    #     self.assertGreater(accuracy, 0.5)
    def test_separation_metrics(self):
        """Test source separation quality using SNR and SDR metrics"""
        # Arrange
        true_sources = TestDataGenerator.generate_multisin_sources(
            self.n_samples,
            frequencies=[10, 20],
            noise_type='gaussian',
            snr=20.0  # High SNR for clean sources
        )

        # Mix sources
        mixing_matrix = tf.constant([
            [0.8, 0.2],
            [0.3, 0.7]
        ], dtype=tf.float32)
        mixed_signals = TestDataGenerator.mix_sources(true_sources, mixing_matrix)

        # Get sources before training
        sources_pre = self.model.ica.get_sources(mixed_signals, state=0)

        # Get initial metrics
        initial_snr = signal_to_noise_ratio(true_sources, sources_pre)
        initial_sdr = signal_to_distortion_ratio(true_sources, sources_pre)

        # Act
        self.model.train(mixed_signals, hmm_max_iter=10, ica_max_iter=2)

        # Get separated sources after training
        separated_sources = self.model.ica.get_sources(mixed_signals, state=0)

        # Assert
        final_snr = signal_to_noise_ratio(true_sources, separated_sources)
        final_sdr = signal_to_distortion_ratio(true_sources, separated_sources)

        # Print metrics for inspection
        print(f"Initial SNR: {initial_snr.numpy()}, Final SNR: {final_snr.numpy()}")
        print(f"Initial SDR: {initial_sdr}, Final SDR: {final_sdr}")

        # Assert improvements
        self.assertGreater(
            tf.reduce_mean(final_snr),
            tf.reduce_mean(initial_snr),
            "SNR should improve after training"
        )

        self.assertGreater(
            tf.reduce_mean(final_sdr),
            tf.reduce_mean(initial_sdr),
            "SDR should improve after training"
        )


class TestHMICASklearn(unittest.TestCase):
    def setUp(self):
        """Setup test data and model"""
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

        # Generate synthetic test data
        self.n_samples = 200
        self.n_features = 2
        self.n_states = 2
        self.n_sources = 2

        # Create two different mixing matrices for different states
        theta1 = np.pi / 4  # 45 degrees
        theta2 = np.pi / 3  # 60 degrees
        self.mixing_matrix1 = np.array([
            [np.cos(theta1), -np.sin(theta1)],
            [np.sin(theta1), np.cos(theta1)]
        ])
        self.mixing_matrix2 = np.array([
            [np.cos(theta2), -np.sin(theta2)],
            [np.sin(theta2), np.cos(theta2)]
        ])

        # Generate source signals
        t = np.linspace(0, 10, self.n_samples)
        source1 = np.sin(2 * np.pi * 2 * t)  # 2 Hz sine wave
        source2 = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
        self.sources = np.column_stack([source1, source2])

        # Mix sources with different matrices for each half of the samples
        self.X = np.zeros((self.n_samples, self.n_features))
        mid_point = self.n_samples // 2
        self.X[:mid_point] = self.sources[:mid_point] @ self.mixing_matrix1.T
        self.X[mid_point:] = self.sources[mid_point:] @ self.mixing_matrix2.T

    def test_init(self):
        """Test model initialization"""
        # Arrange
        model = HMICA(
            n_states=self.n_states,
            n_sources=self.n_sources,
            whiten=True,
            use_gar=True,
            gar_order=2,
            max_iter={'hmm': 5, 'ica': 10},
            random_state=42
        )
        # Act & Assert
        self.assertEqual(model.n_states, self.n_states)
        self.assertEqual(model.n_sources, self.n_sources)
        self.assertTrue(model.whiten)
        self.assertTrue(model.use_gar)
        self.assertEqual(model.gar_order, 2)

    def test_not_fitted(self):
        """Test behavior before fitting"""
        # Arrange
        model = HMICA(
            n_states=self.n_states,
            n_sources=self.n_sources,
            whiten=True,
            use_gar=True,
            gar_order=2,
            max_iter={'hmm': 5, 'ica': 10},
            random_state=42
        )
        X_test = np.random.randn(10, self.n_features)

        # Act & Assert
        self.assertFalse(hasattr(model, 'model_'))
        self.assertFalse(hasattr(model, 'n_features_in_'))
        self.assertFalse(hasattr(model, 'training_history_'))

        with self.assertRaises(NotFittedError):
            model.transform(X_test)
        with self.assertRaises(NotFittedError):
            model.inverse_transform(X_test, np.zeros((10, self.n_states)))

    def test_fit(self):
        """Test model fitting"""
        # Arrange
        model = HMICA(
            n_states=self.n_states,
            n_sources=self.n_sources,
            whiten=True,
            use_gar=True,
            gar_order=2,
            max_iter={'hmm': 5, 'ica': 10},
            random_state=42
        )
        # Act
        model.fit(self.X)
        # Assert
        self.assertIsNotNone(model.model_)
        self.assertIsNotNone(model.training_history_)
        self.assertEqual(model.n_features_in_, self.n_features)

    def test_transform(self):
        """Test source separation"""
        # Arrange
        model = HMICA(
            n_states=self.n_states,
            n_sources=self.n_sources,
            whiten=True,
            use_gar=True,
            gar_order=2,
            max_iter={'hmm': 5, 'ica': 10},
            random_state=42
        )
        model.fit(self.X)

        # Act
        result = model.transform(self.X)

        # Assert
        self.assertIn('source_signals', result)
        self.assertIn('state_sequence', result)

        # Check shapes
        source_signals = result['source_signals']
        state_sequence = result['state_sequence']
        self.assertEqual(source_signals.shape, (self.n_samples, self.n_sources))
        self.assertEqual(state_sequence.shape, (self.n_samples, self.n_states))

        # Check state sequence is one-hot encoded
        self.assertTrue(np.allclose(np.sum(state_sequence, axis=1), 1.0))

    def test_inverse_transform(self):
        """Test reconstruction"""
        # Arrange
        model = HMICA(
            n_states=self.n_states,
            n_sources=self.n_sources,
            whiten=True,
            use_gar=True,
            gar_order=2,
            max_iter={'hmm': 5, 'ica': 10},
            random_state=42
        )
        # Fit model and get sources
        result = model.fit_transform(self.X)

        # Act
        X_reconstructed = model.inverse_transform(
            result['source_signals'],
            result['state_sequence']
        )

        # Assert
        self.assertEqual(X_reconstructed.shape, self.X.shape)

        # Check reconstruction error (should be relatively small)
        reconstruction_error = np.mean((self.X - X_reconstructed) ** 2)
        self.assertLess(reconstruction_error, 0.1)

    def test_fit_transform(self):
        """Test combined fit and transform"""
        # Arrange
        model = HMICA(
            n_states=self.n_states,
            n_sources=self.n_sources,
            whiten=True,
            use_gar=True,
            gar_order=2,
            max_iter={'hmm': 5, 'ica': 10},
            random_state=42
        )
        # Act
        result = model.fit_transform(self.X)
        # Assert
        self.assertIn('source_signals', result)
        self.assertIn('state_sequence', result)
        self.assertEqual(result['source_signals'].shape, (self.n_samples, self.n_sources))
        self.assertEqual(result['state_sequence'].shape, (self.n_samples, self.n_states))

    def test_score(self):
        """Test model scoring"""
        # Arrange
        model = HMICA(
            n_states=self.n_states,
            n_sources=self.n_sources,
            whiten=True,
            use_gar=True,
            gar_order=2,
            max_iter={'hmm': 5, 'ica': 10},
            random_state=42
        )
        model.fit(self.X)
        # Act
        score = model.score(self.X)
        # Assert
        self.assertIsInstance(score, float)

        # Score should be higher for training data than random data
        random_data = np.random.randn(*self.X.shape)
        random_score = model.score(random_data)
        self.assertGreater(score, random_score)

if __name__ == '__main__':
    unittest.main()
