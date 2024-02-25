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
from functools import partial, update_wrapper
import matplotlib.pyplot as plt
from graphviz import Digraph

# exports
__all__ = ['cartesian_to_spherical', 'butter_bandpass_filter', 'moving_average', 
           'window_sinc_convolve', 'acceleration_to_velocity', 'normalize', 
           'hilbert_transform', 'spectrogram', 'noise_remove_ica', 
           'WrappedPartial', 'FunctionNode']


def _cartesian_to_spherical(x, y, z):
    '''
    Given a 3D vector in cartesian coordinates, convert to spherical coordinates

    Args:
        - x, y, z (float):
            x, y, z coordinates of the vector
    
    Returns:
        - r, theta, phi: float
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
        - data (numpy array):
            2D array of (3, n_timepoints) in cartesian coordinates

    Returns:
        - r, theta, phi (numpy array):
            2D array of (3, n_timepoints) in spherical coordinates
    '''
    r, theta, phi = np.apply_along_axis(lambda m: _cartesian_to_spherical(*m), 
                                        axis=0, arr=data)
    return np.vstack((r, theta, phi))


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''
    Perform bandpass filtering on a 2D array of (n_channels, n_timepoints).

    Args:
        - data (numpy array):
            2D array of (n_channels, n_timepoints)
        - lowcut (float):
            Lower cutoff frequency
        - highcut (float):
            Upper cutoff frequency
        - fs (float):
            Sampling frequency
        - order (int):
            Order of the filter
    
    Returns:
        - data: numpy array of (n_channels, n_timepoints)
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
        - data (numpy array):
            2D array of (n_channels, n_timepoints)
        - window_size (int):
            Size of the moving average window

    Returns:
        - data: numpy array of (n_channels, n_timepoints)
    '''
    kernel = np.ones(window_size) / window_size
    data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 
                               axis=1, arr=data)
    return data


def window_sinc_convolve(data, window_size, fs):
    '''
    Perform windowed sinc convolution on a 2D array of (n_channels, n_timepoints).

    Args:
        - data (numpy array):
            2D array of (n_channels, n_timepoints)
        - window_size (int):
            Size of the window
        - fs (float):
            Sampling frequency

    Returns:
        - data: numpy array of (n_channels, n_timepoints)
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
        - data (numpy array):
            2D array of (n_channels, n_timepoints) in acceleration
        - fs (float):
            Sampling frequency

    Returns:
        - data: numpy array of (n_channels, n_timepoints) in velocity
    '''
    data = np.apply_along_axis(lambda m: np.cumsum(m) / fs, axis=1, arr=data)
    return data


def normalize(data):
    '''
    Given a 2D array of (n_channels, n_timepoints), normalize each channel.

    Args:
        - data (numpy array):
            2D array of (n_channels, n_timepoints)

    Returns:
        - data: numpy array of (n_channels, n_timepoints)
    '''
    data = np.apply_along_axis(lambda m: (m - np.mean(m)) / np.std(m), axis=1, arr=data)
    return data


def hilbert_transform(data):
    '''
    Given a 2D array of (n_channels, n_timepoints), perform hilbert transform 
    on each channel and return 2 arrays of (n_channels, n_timepoints) for
    amplitude and phase.

    Args:
        - data (numpy array):
            2D array of (n_channels, n_timepoints)

    Returns:
        - amplitude: numpy array of (n_channels, n_timepoints)
        - phase: numpy array of (n_channels, n_timepoints)
    '''
    hilbert_transformed = np.apply_along_axis(lambda m: hilbert(m), axis=1, arr=data)
    amplitude = np.abs(hilbert_transformed)
    phase = np.angle(hilbert_transformed)
    return amplitude, phase


def noise_remove_ica(data, n_components=3):
    '''
    Given a 2D array of (n_channels, n_timepoints), perform ICA to remove noise.

    Args:
        - data (numpy array):
            2D array of (n_channels, n_timepoints)
        - n_components (int):
            Number of components to keep

    Returns:
        - data: numpy array of (n_channels, n_timepoints)
    '''
    ica = FastICA(n_components=n_components)
    data = ica.fit_transform(data.T).T
    return data
    
    
class WrappedPartial:
    """
    A class representing a wrapped partial function.

    Parameters:
    - original_func (callable): The original function to create a partial function from.
    - *args: Positional arguments to fix in the partial function.
    - **kwargs: Keyword arguments to fix in the partial function.
    """

    def __init__(self, original_func, *args, **kwargs):
        """
        Initialize a WrappedPartial instance.

        Parameters:
        - original_func (callable): The original function to create a partial function from.
        - *args: Positional arguments to fix in the partial function.
        - **kwargs: Keyword arguments to fix in the partial function.
        """
        partial_func = partial(original_func, *args, **kwargs)

        # Preserve __doc__ and __name__ attributes
        update_wrapper(partial_func, original_func)
        self.partial_func = partial_func

    def getfunc(self, new_name=None):
        """
        Get the partial function.

        Parameters:
        - new_name (str, optional): If provided, set the __name__ attribute of the partial function to this value.

        Returns:
        - callable: The partial function.
        """
        partial_func = self.partial_func
        if new_name is not None:
            partial_func.__name__ = new_name
        return partial_func 
    
    
    
class FunctionNode:
    """
    A class representing a node associated with a function.

    Parameters:
    - func (callable): The function associated with the root node.
    - children (list, optional): List of child nodes. Defaults to an empty list.
    """

    def __init__(self, func, children=None):
        """
        Initialize a FunctionTree instance.

        Parameters:
        - func (callable): The function associated with the root node.
        - children (list, optional): List of child nodes. Defaults to an empty list.
        """
        self.func = func
        self.children = children or []

    def add_child(self, child_node):
        """
        Add a child node to the root node.

        Parameters:
        - child_node (FunctionTree): The child node to be added.
        """
        self.children.append(FunctionNode(child_node))

    def find_node(self, target_func):
        """
        Recursively search for a node with a specific function in the tree.

        Parameters:
        - target_func (callable): The target function to search for.

        Returns:
        - FunctionTree or None: The node with the target function, or None if not found.
        """
        if self.func == target_func:
            return self
        else:
            for child in self.children:
                found_node = child.find_node(target_func)
                if found_node:
                    return found_node
        return None

    def add_child_to_node(self, target_func, new_child):
        """
        Add a child node to an arbitrary node in the tree.

        Parameters:
        - target_func (callable): The function associated with the target node.
        - new_child (FunctionTree): The child node to be added to the target node.
        """
        target_node = self.find_node(target_func)
        if target_node:
            target_node.add_child(new_child)
        else:
            print(f"Node with function {target_func} not found.")

    def _evaluate(self, input_value):
        """
        Recursively evaluate the tree and return a list of outputs from all leaf nodes.

        Parameters:
        - input_value: The input value to be used in the function evaluations.

        Returns:
        - list: A list of outputs from all leaf nodes.
        """
        if not self.children:
            return self.func(input_value), 
        else:
            curr_result = self.func(input_value)
            child_results = [child.evaluate(curr_result) for child in self.children]
            
            return  [result for sublist in child_results for result in sublist]
        
    def evaluate(self, input_value):
        """
        Evaluate the tree and return a list of outputs from all leaf nodes.

        Parameters:
        - input_value: The input value to be used in the function evaluations.

        Returns:
        - numpy.ndarray: A numpy array stacking all data produced from the leaf nodes on axis 0.
        The resulting shape is (n_leaves, ...).
        """
        results = self._evaluate(input_value)
        return np.array(results)
        

    def visualize(self, graph=None, parent_name=None, graphviz=None, size=None):
        """
        Generate a graphical representation of the tree using Graphviz.

        Parameters:
        - graph (Digraph, optional): The Graphviz graph. Defaults to None.
        - parent_name (str, optional): The name of the parent node. Defaults to None.
        - graphviz (Digraph, optional): The original Graphviz graph. Defaults to None.
        - size (tuple, optional): The size of the output graph. Defaults to None.

        Returns:
        - Digraph: The Graphviz graph.
        """
        if graph is None:
            graph = Digraph(format='png')

            # Set the size if provided
            if size:
                graph.attr(size=size)

            graphviz = graph
        current_name = str(id(self))
        graph.node(current_name, label=str(self.func.__name__))

        if parent_name is not None:
            graph.edge(parent_name, current_name)

        for i, child in enumerate(self.children):
            child.visualize(graph, current_name, graphviz=graphviz, size=size)

        return graph
    