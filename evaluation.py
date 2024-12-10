"""
-------------------------------------------------------
Implementation of the evaluation functions (signal-to-noise ratio,
signal-to-distortion ration, etc.).
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "12/6/24"
-------------------------------------------------------
"""
from typing import Optional

# Imports
import tensorflow as tf
import numpy as np
import itertools
from fast_bss_eval import si_sdr as sdr


# Constants

def power_norm(x, axis=None) -> tf.Tensor:
    """
    -------------------------------------------------------
    Normalizes the signal to have unit power while preserving relative scales.
    -------------------------------------------------------
    Parameters:
       x - Input tensor (tf.Tensor)
       axis - Axis to compute the mean and standard deviation (int)
    Returns:
        Normalized tensor (tf.Tensor)
    -------------------------------------------------------
    """
    # Compute signal power
    power = tf.reduce_mean(tf.square(x), axis=axis, keepdims=True)
    # Normalize to unit power
    return x / tf.sqrt(power + 1e-8)


def signal_to_noise_ratio(original: tf.Tensor, separated: tf.Tensor) -> tf.Tensor:
    """
    -------------------------------------------------------
    Computes Scale-Invariant Signal-to-Noise Ratio (SNR) in decibels.
    SNR = 10 * log10(signal_power / noise_power)
    https://tinyurl.com/5dv8bmxd
    -------------------------------------------------------
    Parameters:
        original: Clean source signals (tf.Tensor [time_steps, n_sources])
        separated: Estimated source signals (tf.Tensor [time_steps, n_sources])
    Returns:
        signal-to-noise ratio (tf.Tensor [n_sources])
    -------------------------------------------------------
    """
    original_norm = power_norm(original, axis=0)
    separated_norm = power_norm(separated, axis=0)

    # Find optimal scaling factor between original and separated signals
    scale = tf.reduce_sum(original_norm * separated_norm, axis=0) / \
            tf.reduce_sum(tf.square(separated_norm), axis=0)

    # Scale separated signals
    separated_scaled = separated_norm * tf.expand_dims(scale, 0)

    signal_power = tf.reduce_mean(tf.square(original_norm), axis=0)
    noise = original_norm - separated_scaled
    noise_power = tf.reduce_mean(tf.square(noise), axis=0)

    # Avoid division by zero and log(0)
    snr = 10 * (tf.math.log(signal_power/noise_power)) / tf.math.log(10.0)
    return snr


def si_snr(original: tf.Tensor, separated: tf.Tensor, return_perm: Optional[bool] = False):
    """
    -------------------------------------------------------
    Computes Scale-Invariant Signal-to-Noise Ratio (SNR) with permuation search
    -------------------------------------------------------
    Parameters:
       original - Clean source signals (tf.Tensor [time_steps, n_sources])
       separated - Estimated source signals (tf.Tensor [time_steps, n_sources])
       return_perm - Return the permutation that maximizes the SIR (bool)
    Returns:
         signal-to-noise ratio (tf.Tensor [n_sources])
         perm - Permutation that maximizes the SNR (List)
    -------------------------------------------------------
    """
    indices = tf.range(separated.shape[1])
    perms = list(itertools.permutations(indices))
    ssnr = []
    snr = []
    for perm in perms:
        s = tf.gather(separated, perm, axis=1)
        score = signal_to_noise_ratio(original, s)
        snr.append(score)
        ssnr.append(tf.reduce_sum(score))

    max_index = np.argmax(ssnr)
    if return_perm:
        return snr[max_index], perms[max_index]
    return snr[max_index]


def signal_to_distortion_ratio(original: tf.Tensor, separated: tf.Tensor) -> tf.Tensor:
    """
    -------------------------------------------------------
    Computes Scale Invariant Signal-to-Distortioin Ratio (SNR) in decibels.
    -------------------------------------------------------
    Parameters:
        original: Clean source signals (tf.Tensor [time_steps, n_sources])
        separated: Estimated source signals (tf.Tensor [time_steps, n_sources])
    Returns:
        signal-to-noise ratio (tf.Tensor [n_sources])
    -------------------------------------------------------
    """
    val, _ = sdr(tf.transpose(original).numpy(), tf.transpose(separated).numpy(), return_perm=True)
    if not isinstance(val, tf.Tensor):
        val = tf.convert_to_tensor(val, dtype=tf.float32)
    return val
