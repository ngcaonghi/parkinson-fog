"""
Author: nghi
This module contains utility functions for data processing and visualization.

Functions:
    - generate_shorter_sequences: Generates shorter sequences from longer input sequences.
    - spectrogram: Given a 2D array of (n_channels, n_timepoints), perform STFT on each channel
    and return an array of frequencies, an array of time points, and a 3D array 
    of (n_channels, n_freqs, n_timepoints).
    - plot_spectrogram: Plots the spectrogram of a signal.
"""
# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import tensorflow as tf
import os
from pathlib import Path


# exports
__all__ = ['get_filename_no_ext',
           'generate_shorter_sequences', 
           'spectrogram', 
           'plot_spectrogram', 
           'create_template', 
           'populate_submission']

def get_filename_no_ext(file_path):
    """
    Extracts the filename without extension from the given file path.

    Args:
    - file_path: str or Path
        The file path from which the filename without extension will be extracted.

    Returns:
    - filename: str
        The filename without extension.

    Raises:
    - ValueError: If the input file_path is not a string or Path object.

    Example:
    >>> get_filename_no_ext('/path/to/file.txt')
    'file'
    >>> get_filename_no_ext(Path('/path/to/file.txt'))
    'file'
    """
    if isinstance(file_path, str):
        base = os.path.basename(file_path)
    elif isinstance(file_path, Path):
        base = file_path.name
    else:
        raise ValueError("Unsupported file path type. Please provide a string or PosixPath object.")
    
    filename = os.path.splitext(base)[0]
    return filename

def generate_shorter_sequences(long_x, long_y, window_size, stride):
    '''
    Generates shorter sequences from longer input sequences.

    Args:
        - long_x: A numpy array representing the long input sequences with shape (n_samples, n_features).
        - long_y: A numpy array representing the corresponding long target sequences with shape (n_samples, n_targets).
        - window_size: An integer specifying the size of the shorter sequences to be generated.
        - stride: An integer specifying the stride or step size along the long sequences.

    Yields:
        A generator that produces tuples of shorter input and target sequences.

    Example:
        # Generate shorter sequences with window size 5 and stride 2
        long_input = np.array([[1, 2, 3, 4, 5],
                               [6, 7, 8, 9, 10],
                               [11, 12, 13, 14, 15]])
        long_target = np.array([[0, 1],
                                [1, 0],
                                [0, 1]])
        for short_x, short_y in generate_shorter_sequences(long_input, long_target, window_size=3, stride=2):
            print("Short Input:", short_x)
            print("Short Target:", short_y)
    '''
    seq_len = long_x.shape[0]
    for i in range(0, long_x.shape[0] - window_size + 1, stride):
        yield long_x[i:i+window_size, :], long_y[i:i+window_size, :]
    if (seq_len % window_size) % stride > 0 :
        yield long_x[-window_size:, :], long_y[-window_size:, :]


def spectrogram(data, fs, nperseg=128, noverlap=64, max_freq=50, min_freq=0):
    '''
    Given a 2D array of (n_channels, n_timepoints), perform STFT on each channel
    and return an array of frequencies, an array of time points, and a 3D array 
    of (n_channels, n_freqs, n_timepoints).

    Args:
        - data (numpy array):
            2D array of (n_channels, n_timepoints)
        - fs (float):
            Sampling frequency
        - nperseg (int):
            Length of each segment
        - noverlap (int):
            Number of points to overlap between segments
        - max_freq (float):
            Maximum frequency to return
        - min_freq (float):
            Minimum frequency to return

    Returns:
        - frequencies: numpy array of (n_freqs,)
        - timepoints: numpy array of (n_timepoints,)
        - spectrogram: numpy array of (n_channels, n_freqs, n_timepoints)
    '''
    frequencies, timepoints, spectrogram = signal.spectrogram(data, fs=fs, 
                                                              nperseg=nperseg, 
                                                              noverlap=noverlap)
    # Remove frequencies above max_freq and below min_freq
    frequencies = frequencies[(frequencies <= max_freq) & (frequencies >= min_freq)]
    spectrogram = spectrogram[:, (frequencies <= max_freq) & (frequencies >= min_freq), :]
    return frequencies, timepoints, spectrogram
    

def plot_spectrogram(X, fs, maxfreq=5, minfreq=0, nperseg=512, ax=None, figsize=(12, 4)):
    """
    Plots the spectrogram of a signal.

    Args:
        - X (array_like): The input signal.
        fs (float): The sampling frequency of the signal.
        - maxfreq (float, optional): Maximum frequency to display in the spectrogram.
    Default is 5 Hz.
        - minfreq (float, optional): Minimum frequency to display in the spectrogram.
    Default is 0 Hz.
        - nperseg (int, optional): Length of each segment used to compute the FFT.
    Default is 512.
        - ax (matplotlib.axes.Axes, optional): The Axes object to plot on. If None,
    a new figure will be created.
        - figsize (tuple, optional): Figure size (width, height) in inches.
    Default is (12, 4).

    Returns:
        - fig (matplotlib.figure.Figure): The created figure.
        - ax (matplotlib.axes.Axes): The created axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    f, t, Sxx = spectrogram(X, fs, nperseg=nperseg, max_freq=maxfreq, min_freq=minfreq)
    ax.pcolormesh(t, f, Sxx, cmap='magma')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_ylim(minfreq, maxfreq)
    ax.set_title('Spectrogram')
    return fig, ax


def create_template(ids, sample_submission):
    """
    Create a template DataFrame with zeros for the given IDs and columns from the sample_submission DataFrame.

    Args:
        ids (list): List of IDs for the template DataFrame.
        sample_submission (pd.DataFrame): DataFrame containing the columns for the template.

    Returns:
        pd.DataFrame: Template DataFrame with zeros and the given IDs.
    """
    template = pd.DataFrame(np.zeros((len(ids), 4)), columns=sample_submission.columns)
    template['Id'] = ids
    return template


def populate_submission(model, test_dataset, template):
    """
    Populates the submission template with predictions generated by the model.

    Args:
        model (object): The trained model used for prediction.
        test_dataset (object): The dataset used for testing.
        template (object): The submission template.

    Returns:
        object: The populated submission template.
    """
    for batch, id_batch in test_dataset:
        pred = model.predict(batch, verbose=1, use_multiprocessing=True)
        for seqi, id_single in enumerate(id_batch):
            first_id = id_single[0]
            first_id = tf.compat.as_str_any(first_id.numpy()[0])
            last_id = id_single[-1]
            last_id = tf.compat.as_str_any(last_id.numpy()[0])
            
            first_loc = template.loc[template['Id']==first_id].index.tolist()[0]
            last_loc = template.loc[template['Id']==last_id].index.tolist()[0]
              
            template.iloc[first_loc:last_loc+1, 1:] = pred[seqi, :, :-1]
    return template