'''
Author: nghi

Utility functions to preprocess and extract features from the Parkinson's Disease 
freezing-of-gait data, including:
    - Cartesian to spherical coordinate conversion
    - Bandpass filtering
    - Moving average
    - Windowed sinc convolution
    - Acceleration to velocity conversion
    - Normalization
    - Hilbert transform
    - Spectrogram
    - Noise removal using ICA
'''

# imports
import numpy as np
from scipy import signal
from scipy.signal import hilbert
from sklearn.decomposition import FastICA

# exports
__all__ = ['cartesian_to_spherical', 'butter_bandpass_filter', 'moving_average', 
           'window_sinc_convolve', 'acceleration_to_velocity', 'normalize', 
           'hilbert_transform', 'spectrogram', 'noise_remove_ica']


def _cartesian_to_spherical(x, y, z):
    '''
    Given a 3D vector in cartesian coordinates, convert to spherical coordinates

    Args:
        x, y, z: float
            x, y, z coordinates of the vector
    
    Returns:
        r, theta, phi: float
    '''
    r = np.sqrt(x**2 + y**2 + z**2)  # Radial distance
    if r == 0:
        theta = 0  # Avoid division by zero
    else:
        theta = np.arccos(z / r)  # Polar angle
    phi = np.arctan2(y, x)  # Azimuth angle
    return r, theta, phi


def cartesian_to_spherical(data):
    '''
    Given a 2D array of (3, n_timepoints) in cartesian coordinates, 
    convert cartesian coordinates to spherical coordinates.

    Args:
        data: numpy array
            2D array of (3, n_timepoints) in cartesian coordinates

    Returns:
        r, theta, phi: numpy array
            2D array of (3, n_timepoints) in spherical coordinates
    '''
    r, theta, phi = np.apply_along_axis(lambda m: _cartesian_to_spherical(*m), 
                                        axis=0, arr=data)
    return np.vstack((r, theta, phi))


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''
    Perform bandpass filtering on a 2D array of (n_channels, n_timepoints).

    Args:
        data: numpy array
            2D array of (n_channels, n_timepoints)
        lowcut: float
            Lower cutoff frequency
        highcut: float
            Upper cutoff frequency
        fs: float
            Sampling frequency
        order: int
            Order of the filter
    
    Returns:
        data: numpy array of (n_channels, n_timepoints)
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Get the filter coefficients
    b, a = signal.butter(order, [low, high], btype='band')
    data = np.apply_along_axis(lambda m: signal.lfilter(b, a, m), axis=1, arr=data)
    return data


def moving_average(data, window_size):
    '''
    Perform moving average on a 2D array of (n_channels, n_timepoints).

    Args:
        data: numpy array
            2D array of (n_channels, n_timepoints)
        window_size: int
            Size of the moving average window

    Returns:
        data: numpy array of (n_channels, n_timepoints)
    '''
    kernel = np.ones(window_size) / window_size
    data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 
                               axis=1, arr=data)
    return data


def window_sinc_convolve(data, window_size, fs):
    '''
    Perform windowed sinc convolution on a 2D array of (n_channels, n_timepoints).

    Args:
        data: numpy array
            2D array of (n_channels, n_timepoints)
        window_size: int
            Size of the window
        fs: float
            Sampling frequency

    Returns:
        data: numpy array of (n_channels, n_timepoints)
    '''
    kernel = np.sinc(2 * window_size * np.arange(fs) / fs)
    data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 
                               axis=1, arr=data)
    return data


def acceleration_to_velocity(data, fs):
    '''
    Given a 2D array of (n_channels, n_timepoints) in acceleration, 
    convert to velocity.

    Args:
        data: numpy array
            2D array of (n_channels, n_timepoints) in acceleration
        fs: float
            Sampling frequency

    Returns:
        data: numpy array of (n_channels, n_timepoints) in velocity
    '''
    data = np.apply_along_axis(lambda m: np.cumsum(m) / fs, axis=1, arr=data)
    return data


def normalize(data):
    '''
    Given a 2D array of (n_channels, n_timepoints), normalize each channel.

    Args:
        data: numpy array
            2D array of (n_channels, n_timepoints)

    Returns:
        data: numpy array of (n_channels, n_timepoints)
    '''
    data = np.apply_along_axis(lambda m: (m - np.mean(m)) / np.std(m), axis=1, arr=data)
    return data


def hilbert_transform(data):
    '''
    Given a 2D array of (n_channels, n_timepoints), perform hilbert transform 
    on each channel and return 2 arrays of (n_channels, n_timepoints) for
    amplitude and phase.

    Args:
        data: numpy array
            2D array of (n_channels, n_timepoints)

    Returns:
        amplitude: numpy array of (n_channels, n_timepoints)
        phase: numpy array of (n_channels, n_timepoints)
    '''
    hilbert_transformed = np.apply_along_axis(lambda m: hilbert(m), axis=1, arr=data)
    amplitude = np.abs(hilbert_transformed)
    phase = np.angle(hilbert_transformed)
    return amplitude, phase


def spectrogram(data, fs, nperseg=128, noverlap=64, max_freq=50, min_freq=0):
    '''
    Given a 2D array of (n_channels, n_timepoints), perform STFT on each channel
    and return an array of frequencies, an array of time points, and a 3D array 
    of (n_channels, n_freqs, n_timepoints).

    Args:
        data: numpy array
            2D array of (n_channels, n_timepoints)
        fs: float
            Sampling frequency
        nperseg: int
            Length of each segment
        noverlap: int
            Number of points to overlap between segments
        max_freq: float
            Maximum frequency to return
        min_freq: float
            Minimum frequency to return

    Returns:
        frequencies: numpy array of (n_freqs,)
        timepoints: numpy array of (n_timepoints,)
        spectrogram: numpy array of (n_channels, n_freqs, n_timepoints)
    '''
    frequencies, timepoints, spectrogram = signal.spectrogram(data, fs=fs, 
                                                              nperseg=nperseg, 
                                                              noverlap=noverlap)
    # Remove frequencies above max_freq and below min_freq
    frequencies = frequencies[(frequencies <= max_freq) & (frequencies >= min_freq)]
    spectrogram = spectrogram[:, (frequencies <= max_freq) & (frequencies >= min_freq), :]
    return frequencies, timepoints, spectrogram


def noise_remove_ica(data, n_components=3):
    '''
    Given a 2D array of (n_channels, n_timepoints), perform ICA to remove noise.

    Args:
        data: numpy array
            2D array of (n_channels, n_timepoints)
        n_components: int
            Number of components to keep

    Returns:
        data: numpy array of (n_channels, n_timepoints)
    '''
    ica = FastICA(n_components=n_components)
    data = ica.fit_transform(data.T).T
    return data
    