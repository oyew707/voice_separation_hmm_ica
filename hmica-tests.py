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
        self.n_samples = 200
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
            gar_order=2
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

        # Check if log likelihood improves
        self.assertGreater(history['hmm_ll'][-1], history['hmm_ll'][0])

    def test_source_separation(self):
        """Test source separation quality"""
        # Train model
        self.model.train(self.mixed_signals, max_iter=10)

        # Get sources for first state
        sources = self.model.ica.get_sources(
            self.mixed_signals[self.true_states == 0],
            state=0
        )

        # Check source dimensions
        self.assertEqual(sources.shape[1], self.n_sources)

        # Check if sources are more independent than mixed signals
        mixed_corr = np.corrcoef(self.mixed_signals[self.true_states == 0].numpy().T)
        source_corr = np.corrcoef(sources.numpy().T)

        # Off-diagonal correlations should be smaller for sources
        mixed_off_diag = np.abs(mixed_corr[0, 1])
        source_off_diag = np.abs(source_corr[0, 1])
        self.assertLess(source_off_diag, mixed_off_diag)

    def test_state_estimation(self):
        """Test state estimation accuracy"""
        # Train model
        self.model.train(self.mixed_signals, max_iter=10)

        # Get state responsibilities
        obs_prob = lambda state, x: self.model.ica.compute_likelihood(
            tf.expand_dims(x, 0),
            state
        )
        alpha_hat, c_t = self.model.hmm.calc_alpha_hat(
            self.mixed_signals,
            obs_prob
        )

        # Get most likely states
        estimated_states = tf.argmax(alpha_hat, axis=1)

        # Compute accuracy (accounting for possible state permutation)
        accuracy = max(
            np.mean(estimated_states == self.true_states),
            np.mean(estimated_states == 1 - self.true_states)
        )

        # Should be better than random guessing
        self.assertGreater(accuracy, 0.5)


if __name__ == '__main__':
    unittest.main()
