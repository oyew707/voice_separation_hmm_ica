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
            learning_rates={'W': 1e-4, 'R': 1e-4, 'beta': 1e-4, 'C': 1e-4}
        )

    def test_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.k, self.n_states)
        self.assertEqual(self.model.m, self.n_sources)
        self.assertEqual(self.model.x_dims, self.x_dims)

    def test_training(self):
        """Test model training"""
        history = self.model.train(self.mixed_signals, max_iter=10)

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
        self.model.train(self.mixed_signals, max_iter=10)

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
        self.assertLess(sp_off_diag, source_off_diag)

    def test_state_estimation(self):
        """Test state estimation accuracy"""
        # Train model
        self.model.train(self.mixed_signals, max_iter=10)

        # Get state responsibilities
        obs_prob = lambda state, x: self.model.ica.compute_likelihood( x, state)
        obs_prob_matrix = self.model.hmm.calc_obs_prob(self.mixed_signals, obs_prob)
        alpha_hat, c_t = self.model.hmm.calc_alpha_hat(
            self.mixed_signals,
            obs_prob_matrix
        )

        # Get most likely states
        estimated_states = tf.argmax(alpha_hat, axis=1).numpy().astype(np.int32)

        # Compute accuracy (accounting for possible state permutation)
        accuracy = max(
            np.mean(estimated_states == self.true_states),
            np.mean(estimated_states == 1 - self.true_states)
        )

        # Should be better than random guessing
        self.assertGreater(accuracy, 0.5)
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
        self.model.train(mixed_signals, max_iter=20)

        # Get separated sources after training
        separated_sources = self.model.ica.get_sources(mixed_signals, state=0)

        # Assert
        final_snr = signal_to_noise_ratio(true_sources, separated_sources)
        final_sdr = signal_to_distortion_ratio(true_sources, separated_sources)

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

        # Print metrics for inspection
        print(f"Initial SNR: {initial_snr.numpy()}, Final SNR: {final_snr.numpy()}")
        print(f"Initial SDR: {initial_sdr.numpy()}, Final SDR: {final_sdr.numpy()}")

if __name__ == '__main__':
    unittest.main()
