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


# Imports
import tensorflow as tf
import numpy as np
from fast_bss_eval import sdr

# Constants


def signal_to_noise_ratio(original: tf.Tensor, separated: tf.Tensor) -> tf.Tensor:
    """
    -------------------------------------------------------
    Computes Signal-to-Noise Ratio (SNR) in decibels.
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
    signal_power = tf.reduce_mean(tf.square(original), axis=0)
    noise = original - separated
    noise_power = tf.reduce_mean(tf.square(noise), axis=0)

    # Avoid division by zero and log(0)
    snr = 10 * (tf.math.log1p(signal_power) - tf.math.log1p(noise_power)) / tf.math.log(10.0)
    return snr


def signal_to_distortion_ratio(original: tf.Tensor, separated: tf.Tensor) -> tf.Tensor:
    """
    -------------------------------------------------------
    Computes Signal-to-Distortioin Ratio (SNR) in decibels.
    -------------------------------------------------------
    Parameters:
        original: Clean source signals (tf.Tensor [time_steps, n_sources])
        separated: Estimated source signals (tf.Tensor [time_steps, n_sources])
    Returns:
        signal-to-noise ratio (tf.Tensor [n_sources])
    -------------------------------------------------------
    """
    return sdr(tf.transpose(original), tf.transpose(separated))

