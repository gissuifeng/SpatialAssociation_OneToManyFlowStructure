import random
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from scipy.signal import savgol_filter


# Create dictionaries related to city information
def get_city_information():
    cityId_loc_dict, cityId_city_dict, city_cityId_dict = {}, {}, {}
    with open("../../dataset/real_dataset/city_id_location.csv") as fr:
        fr.readline()   # Read the header line
        for line in fr:
            attrs = line.strip().split(",")
            city, id, lng, lat = attrs[0], int(attrs[1]), float(attrs[2]), float(attrs[3])
            cityId_loc_dict[id] = (lng, lat)
            cityId_city_dict[id] = city
            city_cityId_dict[city] = id
    return cityId_loc_dict, cityId_city_dict, city_cityId_dict


# Smooth data using Savitzky-Golay filter
def sg_smooth(data, window_length=11, polyorder=3):
    """
    Parameters:
    data (list or np.array): The data sequence to be smoothed
    window_length (int): The size of the filter window, must be odd
    polyorder (int): The order of the polynomial

    Returns:
    smoothed_data (np.array): The smoothed data sequence
    """
    smoothed_data = savgol_filter(data, window_length, polyorder)

    return smoothed_data

# Calculate DTW distance between two curves
def curve_dtw_similarity(curve1, curve2):
    """
    Parameters:
    curve1 (list or np.array): Data points sequence of the first curve
    curve2 (list or np.array): Data points sequence of the second curve

    Returns:
    dist (float): DTW distance, the smaller the value, the higher the similarity
    """
    # Equidistant sampling
    n = min(len(curve1), len(curve2))
    curve1 = np.linspace(0, 1, n) * curve1
    curve2 = np.linspace(0, 1, n) * curve2

    # Calculate DTW distance
    dist = dtw(curve1, curve2)

    return dist

# Calculate p-value based on DTW algorithm using Monte Carlo simulation
def monte_carlo_dtw_test(curve1, curve2, num_samples=1000, significance_level=0.01):
    # Store all sampled DTW distances
    distance_measurements = []
    # Calculate the original DTW distance between the original sequences
    original_distance = curve_dtw_similarity(curve1, curve2)
    # Generate random samples and calculate the DTW distance with the original sequence
    for _ in range(num_samples):
        # Generate a random sequence with the same length
        random_sequence = curve2.copy()
        random.shuffle(random_sequence)
        # Calculate the DTW distance between the random sequence and the original sequence
        distance = curve_dtw_similarity(curve1, random_sequence)
        distance_measurements.append(distance)
    # Calculate the p-value
    p_value = np.mean(np.array(distance_measurements) <= original_distance)
    # Determine whether to accept the null hypothesis
    if p_value < significance_level:
        print(f"At significance level {significance_level}, reject the null hypothesis, the two sequences are similar.")
    else:
        print(f"At significance level {significance_level}, accept the null hypothesis, the two sequences are not similar.")
    return round(original_distance, 2), round(p_value, 3)


# Calculate Pearson coefficient and p-value
def p_value_pearsonr(curve1, curve2, significance_level=0.05):
    # Calculate Pearson coefficient and p-value
    pearson_corr, p_value = pearsonr(curve1, curve2)
    print(f"Pearson coefficient: {pearson_corr:.3f}, p-value: {p_value:.3f}")
    return round(pearson_corr, 3), round(p_value, 3)


# Define a function named plot_curve to accept any number of curves as parameters
def plot_curve(*curves, title=""):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Set Chinese font to SimHei
    plt.rcParams['axes.unicode_minus'] = False  # Fix issue of '-' sign displaying as square in saved images
    # Create a new figure window, set the size to 8 inches wide and 6 inches high
    plt.figure(figsize=(8, 6))
    # Iterate through each curve passed in
    for index, curve in enumerate(curves):
        # Generate x-axis coordinates using np.linspace, sequence length matches the curve data length
        x = np.arange(1, len(curve) + 1)
        # Plot the current curve and add a label 'Curve' + index number
        plt.plot(x, curve, label='Curve' + str(index), marker='o')
    plt.xlabel('X')  # Set x-axis label to 'X'
    plt.ylabel('Y')  # Set y-axis label to 'Y'
    plt.title(str(title) + 'Curve Comparison')  # Set the plot title to 'Curve Comparison'
    plt.legend()  # Display the legend
    plt.show()  # Show the plot


# DTW-based value similarity measure and p-value detection
def curve_dtw_similarity_p_test(curve1, curve2):
    # Smooth the data using Savitzky-Golay filter
    curve1 = sg_smooth(curve1, window_length=7, polyorder=2)
    curve2 = sg_smooth(curve2, window_length=7, polyorder=2)
    # Calculate DTW distance between the two curves
    dtw_similarity = curve_dtw_similarity(curve1, curve2)
    print(f"Value similarity between the two curves (DTW): {dtw_similarity:.4f}")
    # Use Monte Carlo method to test the significance of the similarity between the two sequences
    DTW_distance, p_value = monte_carlo_dtw_test(curve1, curve2)
    return DTW_distance, p_value












