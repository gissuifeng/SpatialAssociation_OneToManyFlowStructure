import random
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from scipy.signal import savgol_filter

# Smooth the data using Savitzky-Golay filter
def sg_smooth(data, window_length=11, polyorder=3):
    """
    Parameters:
    data (list or np.array): Data series to be smoothed
    window_length (int): Filter window size, must be odd
    polyorder (int): Polynomial order

    Returns:
    smoothed_data (np.array): Smoothed data series
    """
    smoothed_data = savgol_filter(data, window_length, polyorder)
    return smoothed_data

# Calculate the DTW distance between two curves
def curve_dtw_similarity(curve1, curve2):
    """
    Parameters:
    curve1 (list or np.array): Data points of the first curve
    curve2 (list or np.array): Data points of the second curve

    Returns:
    dist (float): DTW distance, the smaller the value, the higher the similarity
    """
    # Equidistant sampling
    # n = min(len(curve1), len(curve2))
    # curve1 = np.linspace(0, 1, n) * curve1
    # curve2 = np.linspace(0, 1, n) * curve2

    # Calculate DTW distance
    dist = dtw(curve1, curve2)

    return dist

def fun(dis1, dis_list):
    dis_list.sort()
    print(round(dis1,4),round(dis_list[0],4),round(dis_list[-1]),4)
    for i in range(len(dis_list)-2):
        if dis1 > dis_list[i]:
            print(f"============{i}===============")

# Use Monte Carlo simulation to calculate the p-value of two sequences based on DTW algorithm
def monte_carlo_dtw_test(curve1, curve2, num_samples=1000, significance_level=0.01):
    # Store all DTW distances from sampling
    distance_measurements = []

    # Calculate the DTW distance between the original sequence and the first sequence
    original_distance = curve_dtw_similarity(curve1, curve2)

    # Generate random samples and calculate the DTW distance with the original sequence
    for _ in range(num_samples):
        # Generate random sequence with the same length
        random_sequence = curve2.copy()
        random.shuffle(random_sequence)
        # Calculate the DTW distance between the random sequence and the original sequence
        distance = curve_dtw_similarity(curve1, random_sequence)
        distance_measurements.append(distance)

    # fun(original_distance, distance_measurements)
    # Calculate p-value
    p_value = np.mean(np.array(distance_measurements) <= original_distance)
    print(f"disDTW: {original_distance:.3f}, P_value: {p_value:.3f}")
    # Determine whether to accept the null hypothesis
    if p_value < significance_level:
        print(f"At significance level {significance_level}, reject the null hypothesis, the two sequences are similar.")
    else:
        print(f"At significance level {significance_level}, accept the null hypothesis, the two sequences are not similar.")

# Calculate Pearson correlation coefficient and p-value
def p_value_pearsonr(curve1, curve2, significance_level=0.05):
    # Calculate Pearson correlation coefficient and p-value
    pearson_corr, p_value = pearsonr(curve1, curve2)
    print(f"Pearson Correlation: {pearson_corr:.3f}, P-value: {p_value:.3f}")
    # Determine whether to accept the null hypothesis
    if pearson_corr > 0.2 and abs(p_value) <= significance_level:
        print(f"At significance level {significance_level}, reject the null hypothesis, the two sequences are similar.")
    else:
        print(f"At significance level {significance_level}, accept the null hypothesis, the two sequences are not similar.")

# Define a function named plot_curve that can accept any number of curves as arguments
def plot_curve(*curves):
    # Create a new figure window and set the size to 8 inches wide and 6 inches high
    plt.figure(figsize=(8, 6))
    # Iterate through each passed curve
    # enumerate(curves) returns an enumerate object containing the index and the corresponding curve data
    for index, curve in enumerate(curves):
        # Generate x-axis coordinates sequence using np.linspace function, with the same length as the curve data
        x = np.linspace(0, 1, len(curve))
        # Plot the current curve in the figure and add a label 'Curve' + index number
        plt.plot(x, curve, label='Curve'+str(index), marker='o')
    plt.xlabel('X') # Set x-axis label to 'X'
    plt.ylabel('Y') # Set y-axis label to 'Y'
    plt.title('Curve Comparison') # Set figure title to 'Curve Comparison'
    plt.legend() # Display legend
    plt.show() # Display figure window

# Similarity measurement and p-value detection based on DTW value
def curve_dtw_similarity_p_test(curve1, curve2):
    # Smooth the data using Savitzky-Golay filter
    curve1 = sg_smooth(curve1, window_length=7, polyorder=2)
    curve2 = sg_smooth(curve2, window_length=7, polyorder=2)
    # Calculate the DTW distance between two curves
    dtw_similarity = curve_dtw_similarity(curve1, curve2)
    print(f"DTW similarity of the two curves: {dtw_similarity:.4f}")
    # Test the significance of the similarity of the two sequences using Monte Carlo method
    monte_carlo_dtw_test(curve1, curve2)

# Similarity measurement and p-value detection based on Pearson shape
def curve_person_similarity_p_test(curve1, curve2, significance_level=0.05):
    # Smooth the data using Savitzky-Golay filter
    curve1 = sg_smooth(curve1, window_length=7, polyorder=2)
    curve2 = sg_smooth(curve2, window_length=7, polyorder=2)
    # Calculate Pearson correlation coefficient and p-value
    pearson_corr, p_value = pearsonr(curve1, curve2)
    print(f"Pearson Correlation: {pearson_corr:.3f}, P-value: {p_value:.3f}")
    # Determine whether to accept the null hypothesis
    if pearson_corr > 0.2 and abs(p_value) <= significance_level:
        print(f"At significance level {significance_level}, reject the null hypothesis, the two sequences are similar.")
    else:
        print(f"At significance level {significance_level}, accept the null hypothesis, the two sequences are not similar.")
    return round(pearson_corr, 3), round(p_value, 3)

#========================Synthetic Data Set Shape Similarity==========================================

#=========================Synthetic Data Set: 1-6 rows are value similarity, 7-12 are shape similarity, 13-14 are randomly generated============================================

net1 = [[49, 83, 87, 100, 88, 77, 63, 35, 21, 10, 10, 23, 32, 47],
        [90, 87, 62, 41, 19, 15, 10, 17, 38, 64, 91, 99, 100, 84],
        [94, 97, 87, 60, 44, 19, 13, 10, 17, 41, 62, 89, 94, 100],
        [16, 13, 23, 50, 66, 91, 97, 100, 93, 69, 48, 21, 16, 10],
        [10, 18, 24, 31, 38, 44, 51, 60, 68, 72, 80, 88, 94, 100],
        [100, 88, 82, 79, 70, 61, 60, 51, 46, 39, 31, 23, 16, 10],
        [62, 96, 100, 113, 101, 90, 76, 48, 34, 23, 23, 36, 45, 60],
        [99, 96, 71, 50, 28, 24, 19, 26, 47, 73, 100, 108, 109, 93],
        [101, 104, 94, 67, 51, 26, 20, 17, 24, 48, 69, 96, 101, 107],
        [23, 20, 30, 57, 73, 98, 104, 107, 100, 76, 55, 28, 23, 17],
        [22, 30, 36, 43, 50, 56, 63, 72, 80, 84, 92, 100, 106, 112],
        [105, 93, 87, 84, 75, 66, 65, 56, 51, 44, 36, 28, 21, 15],
        [95, 33, 64, 15, 48, 93, 36, 83, 44, 55, 10, 59, 27, 71],
        [63, 71, 23, 89, 58, 26, 57, 81, 57, 22, 28, 77, 94, 62]]



net2 = [[31, 78, 74, 97, 100, 73, 63, 46, 22, 19, 2, 15, 41, 56],
        [86, 71, 71, 51, 23, 1, 3, 16, 56, 63, 109, 90, 82, 88],
        [79, 95, 81, 60, 38, 29, 24, 2, 17, 46, 78, 96, 82, 105],
        [1, 18, 34, 44, 60, 97, 81, 102, 80, 81, 42, 22, 14, 29],
        [21, 21, 15, 44, 48, 28, 50, 50, 60, 68, 98, 80, 113, 117],
        [109, 94, 96, 69, 78, 80, 78, 50, 36, 24, 17, 26, 3, 4],
        [161, 195, 199, 212, 200, 189, 175, 147, 133, 122, 122, 135, 144, 159],
        [210, 207, 182, 161, 139, 135, 130, 137, 158, 184, 211, 219, 220, 204],
        [208, 211, 201, 174, 158, 133, 127, 124, 131, 155, 176, 203, 208, 214],
        [125, 122, 132, 159, 175, 200, 206, 209, 202, 178, 157, 130, 125, 119],
        [115, 123, 129, 136, 143, 149, 156, 165, 173, 177, 185, 193, 199, 205],
        [208, 196, 190, 187, 178, 169, 168, 159, 154, 147, 139, 131, 124, 118],
        [99, 84, 21, 62, 59, 10, 73, 33, 64, 64, 75, 30, 75, 37],
        [11, 50, 89, 70, 19, 72, 22, 17, 68, 30, 22, 27, 12, 81]]






for i in range(len(net1)):
    curve1 = np.array(net1[i])
    curve2 = np.array(net2[i])
    print(i + 1)
    # Calculate the Pearson similarity of the two curves
    curve_person_similarity_p_test(curve1, curve2)
    # Calculate the DTW similarity of the two curves
    curve_dtw_similarity_p_test(curve1, curve2)
    # Plot all curves
    plot_curve(curve1, curve2)





