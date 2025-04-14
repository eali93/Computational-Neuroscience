"""
This script implements a t-SNE analysis of neural activity data from handwriting tasks.
It includes time-warping to account for variations in writing speed and PCA for dimensionality reduction.
The analysis produces a 2D visualization of character representations and calculates classification accuracy.
"""

import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def gaussSmooth_fast(timeSeries, width):
    """
    Fast Gaussian smoothing implementation for neural activity data.
    
    Parameters:
    -----------
    timeSeries : numpy.ndarray
        The neural activity data to be smoothed
    width : float
        The width of the Gaussian kernel (standard deviation)
        
    Returns:
    --------
    smoothed : numpy.ndarray
        The smoothed neural activity data
    """
    if width == 0:
        return timeSeries

    # Calculate kernel size based on width (5 standard deviations in each direction)
    wingSize = np.ceil(width * 5)
    x_range = np.arange(-wingSize, wingSize + 1)
    
    # Create Gaussian kernel
    gKernel = np.exp(-x_range**2 / (2 * width**2))
    gKernel = gKernel / np.sum(gKernel)  # Normalize to sum to 1

    # Apply the filter using convolution
    smoothed = np.apply_along_axis(
        lambda x: np.convolve(x, gKernel, mode='same'),
        axis=0,
        arr=timeSeries
    )
    return smoothed

def tsne_warp_dist(d1, d2_mat, n_time_bins_per_trial):
    """
    Time-warped distance function for t-SNE that accounts for variations in writing speed.
    
    Parameters:
    -----------
    d1 : numpy.ndarray
        A single data point (neural activity pattern)
    d2_mat : numpy.ndarray
        Matrix of data points to compare against
    n_time_bins_per_trial : int
        Number of time bins in each trial
        
    Returns:
    --------
    warp_dist : numpy.ndarray
        Array of minimum distances between d1 and each point in d2_mat
    """
    # Define range of time warping factors (alpha values)
    affine_warps = np.linspace(0.7, 1.42, 15)  # 15 different warping factors

    # Calculate number of neural dimensions
    n_neural_dim = len(d1) // n_time_bins_per_trial

    # Reshape data into time x neural dimensions
    d1 = d1.reshape(n_time_bins_per_trial, n_neural_dim)

    # Initialize distance matrix
    e_dist = np.zeros((d2_mat.shape[0], len(affine_warps)))

    # Calculate distances for each warping factor
    for a in range(len(affine_warps)):
        # Create warped time points
        x_orig = np.arange(1, d1.shape[0] + 1)
        x_interp = np.linspace(1, d1.shape[0], int(affine_warps[a] * d1.shape[0]))
        
        # Interpolate data at warped time points
        d1_interp = np.interp(x_interp, x_orig, d1, axis=0)

        # Calculate distances for each comparison point
        for row_idx in range(d2_mat.shape[0]):
            d2 = d2_mat[row_idx, :].reshape(n_time_bins_per_trial, n_neural_dim)
            
            # Handle different warping cases
            if affine_warps[a] > 1:
                df = d1_interp[:d1.shape[0], :] - d2
            else:
                df = d1_interp - d2[:d1_interp.shape[0], :]
            
            # Store mean squared difference
            e_dist[row_idx, a] = np.mean(df**2)

    # Take minimum distance across all warping factors
    warp_dist = np.min(e_dist, axis=1)
    return warp_dist

def main():
    """
    Main function that performs the complete analysis pipeline:
    1. Loads and preprocesses neural data
    2. Performs PCA dimensionality reduction
    3. Computes t-SNE visualization
    4. Calculates classification accuracy
    """
    # Load neural activity data from .mat file
    dat = scipy.io.loadmat('t5.2019.05.08_singleLetters.mat')
    
    # Define all characters to analyze
    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
               'greaterThan','comma','apostrophe','tilde','questionMark']

    # Normalize neural activity by blockwise z-scoring
    for letter in letters:
        # Convert to float32 for numerical stability
        norm_cube = np.array(dat[f'neuralActivityCube_{letter}'], dtype=np.float32)
        
        # Process each block of trials
        t_idx = np.arange(3)
        for y in range(9):
            # Create mean and standard deviation arrays
            mn = np.zeros((3, 1, 192))
            mn[0, 0, :] = dat['meansPerBlock'][y, :]
            mn[1, 0, :] = dat['meansPerBlock'][y, :]
            mn[2, 0, :] = dat['meansPerBlock'][y, :]
            
            sd = np.zeros((1, 1, 192))
            sd[0, 0, :] = dat['stdAcrossAllData']
            
            # Apply z-scoring
            norm_cube[t_idx, :, :] -= mn
            norm_cube[t_idx, :, :] /= sd
            t_idx += 3
        
        dat[f'neuralActivityCube_{letter}'] = norm_cube

    # Compute trial-averaged activity for each character
    all_avg = []
    for letter in letters:
        letter_cube = np.array(dat[f'neuralActivityCube_{letter}'])
        # Average across trials and smooth
        avg_let = np.mean(letter_cube, axis=0)
        avg_let = gaussian_filter1d(avg_let[60:, :], 5, axis=0)  # Remove first 60 time points
        all_avg.append(avg_let)

    # Apply PCA to reduce dimensionality
    all_avg = np.vstack(all_avg)
    pca = PCA(n_components=15)  # Keep 15 principal components
    pca.fit(all_avg)

    # Transform individual trials using PCA components
    all_data = np.zeros((2000, 142 * 15))  # 142 time bins * 15 dimensions
    all_labels = np.zeros(2000, dtype=int)
    c_idx = 0

    for f, letter in enumerate(letters):
        letter_cube = np.array(dat[f'neuralActivityCube_{letter}'])
        for x in range(letter_cube.shape[0]):
            # Process each trial
            row = gaussian_filter1d(letter_cube[x, 60:, :], 3, axis=0)
            # Transform using PCA components
            row = (row - pca.mean_) @ pca.components_[:15].T
            all_data[c_idx, :] = row.flatten()
            all_labels[c_idx] = f
            c_idx += 1

    # Clean up data
    all_data = np.nan_to_num(all_data)
    all_data = all_data[:c_idx, :]
    all_labels = all_labels[:c_idx]

    # Compute t-SNE using warp-distance
    n_time_bins_per_point = 142
    warp_dist_fun = lambda d1, d2: tsne_warp_dist(d1, d2, n_time_bins_per_point)
    
    # Calculate pairwise distances
    D = pdist(all_data, warp_dist_fun)
    D = squareform(D)
    
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=40, metric='precomputed', verbose=2)
    Y = tsne.fit_transform(D)

    # Define plotting parameters
    plot_letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                   '>',',',"'",'~','?']

    # Define colors for each character (matching MATLAB colors)
    colors = np.array([
        [0.3613, 0.8000, 0],      # a
        [0.8000, 0, 0.1548],      # b
        [0.8000, 0.1548, 0],      # c
        [0.8000, 0, 0.4645],      # d
        [0.6710, 0, 0.8000],      # e
        [0.3613, 0, 0.8000],      # f
        [0.5161, 0.8000, 0],      # g
        [0.8000, 0.6194, 0],      # h
        [0.8000, 0, 0],           # i
        [0.6710, 0.8000, 0],      # j
        [0.2065, 0, 0.8000],      # k
        [0, 0.1032, 0.8000],      # l
        [0.8000, 0.3097, 0],      # m
        [0, 0.7226, 0.8000],      # n
        [0, 0.8000, 0.5677],      # o
        [0, 0.8000, 0.2581],      # p
        [0, 0.2581, 0.8000],      # q
        [0.0516, 0, 0.8000],      # r
        [0.8000, 0, 0.6194],      # s
        [0.8000, 0.4645, 0],      # t
        [0.8000, 0, 0.7742],      # u
        [0.8000, 0.7742, 0],      # v
        [0, 0.5677, 0.8000],      # w
        [0.8000, 0, 0.3097],      # x
        [0, 0.8000, 0.4129],      # y
        [0.0516, 0.8000, 0],      # z
        [0, 0.4129, 0.8000],      # >
        [0.5161, 0, 0.8000],      # ,
        [0.2065, 0.8000, 0],      # '
        [0, 0.8000, 0.1032],      # ~
        [0, 0.8000, 0.7226]       # ?
    ])

    # Create t-SNE visualization
    plt.figure(figsize=(10, 10))
    for x in range(len(Y)):
        plt.text(Y[x, 0], Y[x, 1], plot_letters[all_labels[x]], 
                color=colors[all_labels[x]], fontweight='bold', fontsize=6)
    
    plt.axis('equal')
    plt.axis('off')
    plt.show()

    # Calculate k-nearest neighbor classification accuracy
    class_acc = np.zeros(len(D))
    for x in range(len(class_acc)):
        # Find 10 nearest neighbors (excluding self)
        sort_idx = np.argsort(D[x, :])[1:11]
        # Predict class based on majority vote
        choice = np.bincount(all_labels[sort_idx]).argmax()
        class_acc[x] = choice == all_labels[x]

    # Print classification results
    print('Warp NN Accuracy:', np.mean(class_acc))
    print('Confidence Interval:', np.percentile(class_acc, [2.5, 97.5]))

if __name__ == "__main__":
    main() 