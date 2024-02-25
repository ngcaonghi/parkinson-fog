from .feature_extraction import *
from .utils import *
from .constants import *
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

# exports
__all__ = ['fog_pipeline', 'prepare_fog_traning', 'prepare_fog_testing']

def fog_pipeline(fs):
    '''
    Construct a function tree for the FOG dataset.

    Args:
        - fs (int):
            Sampling frequency of the data
    
    Returns:
        - preproc_tree (FunctionNode):
            A function tree that can be used to preprocess the FOG dataset
    '''
    # define partial functions so that they only accepts arrays
    bandpass_delta = WrappedPartial(butter_bandpass_filter, lowcut=0.5, highcut=4, fs=fs).getfunc('bandpass_delta')
    bandpass_theta = WrappedPartial(butter_bandpass_filter, lowcut=4, highcut=8, fs=fs).getfunc('bandpass_theta')
    bandpass_alpha = WrappedPartial(butter_bandpass_filter, lowcut=8, highcut=16, fs=fs).getfunc('bandpass_alpha')
    average_20_hz = WrappedPartial(moving_average, window_size=fs//5).getfunc('average_20_hz')
    acceleration = WrappedPartial(lambda x : x).getfunc('acceleration')
    velocity = WrappedPartial(acceleration_to_velocity, fs=fs).getfunc('velocity')

    # define tuple-handing leaf-node functions for hilbert
    amplitude = lambda htup: htup[0] 
    phase = lambda htup: htup[1] 

    # Constructing a function tree, starting with normalization
    preproc_tree = FunctionNode(normalize)

    # Level 1 functions
    preproc_tree.add_child(bandpass_delta)
    preproc_tree.add_child(bandpass_theta)
    preproc_tree.add_child(bandpass_alpha)
    preproc_tree.add_child(average_20_hz)

    # Level 2 functions
    for band, child in zip(['theta', 'alpha'],
                           [bandpass_theta, bandpass_alpha]):   
        hilbert_node = WrappedPartial(hilbert_transform).getfunc('hilbert_'+band)
        preproc_tree.add_child_to_node(child, hilbert_node)
    #     Level 3 functions
        preproc_tree.add_child_to_node(hilbert_node, WrappedPartial(amplitude).getfunc('amplitude_'+band))
        preproc_tree.add_child_to_node(hilbert_node, WrappedPartial(phase).getfunc('phase_'+band))
    preproc_tree.add_child_to_node(average_20_hz,acceleration)
    preproc_tree.add_child_to_node(average_20_hz, cartesian_to_spherical)
    preproc_tree.add_child_to_node(average_20_hz, velocity)
    return preproc_tree


def prepare_fog_traning(sessions_dir, session_ids, fs, window_size, stride):
    """
    Prepare the training data for fog detection.

    Args:
        - sessions_dir (str): The directory path where the session files are located.
        - session_ids (list): List of session IDs to process.
        - fs (int): The sampling frequency of the data.
        - window_size (int): The size of the sliding window for generating shorter sequences.
        - stride (int): The stride length for the sliding window.

    Returns:
        - tuple: A tuple containing the training data (X_train) and the corresponding labels (y_train).
    """
    X_all = []
    y_all = []
    preproc_tree = fog_pipeline(fs=fs)

    for session_id in tqdm(session_ids):
        df = pd.read_csv(sessions_dir / '{}.csv'.format(session_id))
        x_arr = df[feature_cols].to_numpy().T
        y_arr = df[target_cols].to_numpy().T
        if ('Valid' in df.columns) and ('Task' in df.columns):
            mask = np.all(df[['Valid', 'Task']].to_numpy(), axis=1)
            x_arr = x_arr[:, mask]
            y_arr = y_arr[:, mask]
        if np.sum(y_arr) == 0:
            continue
        del df
        gc.collect()
        x_arr = preproc_tree.evaluate(x_arr)
        x_arr = x_arr.reshape((x_arr.shape[0]*x_arr.shape[1], -1))
        X_all.append(x_arr.T)
        del x_arr
        gc.collect()
        y_all.append(y_arr.T)
        del y_arr
        gc.collect()
    
    X_train = []
    y_train = []
    
    for x_long, y_long in zip(X_all, y_all): 
        for x, y in generate_shorter_sequences(x_long, y_long, window_size, stride):
            X_train.append(x)
            y = np.hstack([y, ~np.any(y, axis=1)[..., None]])
            y_train.append(y)
        
    X_train, y_train = np.array(X_train), np.array(y_train)
        
    return X_train, y_train


def prepare_fog_testing(sessions_dir, session_ids, fs, window_size, stride):
    """
    Prepare the data for fog testing.

    Args:
        - sessions_dir (str): The directory path where the session files are located.
        - session_ids (list): List of session IDs.
        - fs (int): The sampling frequency of the data.
        - window_size (int): The size of the sliding window for generating shorter sequences.
        - stride (int): The stride length for generating shorter sequences.

    Returns:
        - tuple: A tuple containing the prepared data for testing. 
        The first element is the input data (X_test) 
        and the second element is the corresponding ID and time information (X_id_time).
    """
    X_all = []
    X_ids = []
    preproc_tree = fog_pipeline(fs=fs)

    for session_id in tqdm(session_ids):
        df = pd.read_csv(sessions_dir / '{}.csv'.format(session_id))
        id_time = [f'{session_id}_{t}' for t in range(len(df))]
        X_ids.append(np.array(id_time)[..., None])
        x_arr = df[feature_cols].to_numpy().T
        if ('Valid' in df.columns) and ('Task' in df.columns):
            mask = np.all(df[['Valid', 'Task']].to_numpy(), axis=1)
            x_arr = x_arr[:, mask]
        del df
        gc.collect()
        x_arr = preproc_tree.evaluate(x_arr)
        x_arr = x_arr.reshape((x_arr.shape[0]*x_arr.shape[1], -1))
        X_all.append(x_arr.T)
        del x_arr
        gc.collect()
    
    X_test = []
    X_id_time = []
    
    for x_long, id_long in zip(X_all, X_ids): 
        for x, i in generate_shorter_sequences(x_long, id_long, window_size, stride):
            X_test.append(x)
            X_id_time.append(i)
              
    X_test, X_id_time = np.array(X_test), np.array(X_id_time)
    if X_test.ndim == 2:
        X_test = X_test[None, ...]
        
    return X_test, X_id_time