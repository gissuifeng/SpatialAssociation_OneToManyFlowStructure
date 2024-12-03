import numpy as np
import tools
import matplotlib.pyplot as plt
import matplotlib



# Read attention flow and migration flow files to create matrices for both flows
# city_num: the number of cities, used to determine the matrix size
def read_attention_move_file(city_num):
    # --------------------- Read attention flow file and create the flow matrix -----------------------
    attention_array = np.zeros((city_num, city_num))
    with open("../../dataset/real_dataset/attention_flow.csv") as fr:    # Attention flow
        fr.readline()   # Skip the header line
        for line in fr:
            attrs = line.strip().split(",")
            start_city, end_city, flow_size = int(attrs[0]), int(attrs[1]), round(float(attrs[2]), 2)
            attention_array[start_city, end_city] = flow_size
    # --------------------- Read migration flow file and create the flow matrix -----------------------
    move_array = np.zeros((city_num, city_num))
    with open("../../dataset/real_dataset/migration_flow.csv") as fr:    # Migration flow
        fr.readline()   # Skip the header line
        for line in fr:
            attrs = line.strip().split(",")
            start_city, end_city, flow_size = int(attrs[0]), int(attrs[1]), round(float(attrs[2]), 2)
            move_array[start_city, end_city] = flow_size
    return attention_array, move_array

# Analyze the one-to-many structural similarity formed separately by the migration flow and attention flow of the same node
def attention_move_pearson_sim_fun():
    with open("../../result/shape_similarity_results/move_attention_shape_similarity_result.csv", 'w') as fw:
        fw.write("cities, longitude, latitude, Pearson coefficient, p-value\n")
        for i in range(city_num):
            # Smooth the curves for migration flow and attention flow using Savitzky-Golay filter
            smooth_move_curve1 = tools.sg_smooth(move_array[i])
            smooth_move_curve_2 = tools.sg_smooth(attention_array[i])
            # Calculate the Pearson correlation coefficient and p-value between the smoothed curves
            pearson_corr, p_value = tools.p_value_pearsonr(smooth_move_curve1, smooth_move_curve_2)
            # Get the longitude and latitude of the city
            lng, lat = cityId_loc_dict[i]
            # Prepare city information: city name, longitude, latitude
            info = f"{cityId_city_dict[i]},{lng},{lat}"
            # Write the information and calculated values to the file
            fw.write(f"{info},{pearson_corr},{p_value}\n")

# For the attention flow data, analyze the one-to-many structural similarity formed separately by the inflow and outflow data of the same node
def attention_in_out_pearson_sim_fun():
    attention_array_T = attention_array.T
    with open("../../result/shape_similarity_results/attention_inflow_outflow_shape_similarity_result.csv", 'w') as fw:
        fw.write("cities, longitude, latitude, Pearson coefficient, p-value\n")
        for i in range(city_num):
            # Smooth the curves for inflow and outflow using Savitzky-Golay filter
            smooth_attention_curve1 = tools.sg_smooth(attention_array_T[i])
            smooth_attention_curve_2 = tools.sg_smooth(attention_array[i])
            # Calculate the Pearson correlation coefficient and p-value between the smoothed curves
            pearson_corr, p_value = tools.p_value_pearsonr(smooth_attention_curve1, smooth_attention_curve_2)
            # Get the longitude and latitude of the city
            lng, lat = cityId_loc_dict[i]
            # Prepare city information: city name, longitude, latitude
            info = f"{cityId_city_dict[i]},{lng},{lat}"
            # Write the information and calculated values to the file
            fw.write(f"{info},{pearson_corr},{p_value}\n")
            if pearson_corr > 0.6:
                print(f"--- {i} ----")

# For the migration flow data, analyze the one-to-many structural similarity formed separately by the inflow and outflow data of the same node
def move_in_out_pearson_sim_fun():
    move_array_T = move_array.T
    with open("../../result/shape_similarity_results/move_inflow_outflow_shape_similarity_result.csv", 'w') as fw:
        fw.write("cities, longitude, latitude, Pearson coefficient, p-value\n")
        for i in range(5):  # Note: currently only processing the first 5 cities for testing
            # Smooth the curves for inflow and outflow using Savitzky-Golay filter
            smooth_move_curve1 = tools.sg_smooth(move_array_T[i])
            smooth_move_curve_2 = tools.sg_smooth(move_array[i])
            # Plot the original and smoothed curves for visual comparison
            tools.plot_curve(move_array_T[i], move_array[i])
            tools.plot_curve(smooth_move_curve1, smooth_move_curve_2)
            # Calculate the Pearson correlation coefficient and p-value between the smoothed curves
            pearson_corr, p_value = tools.p_value_pearsonr(smooth_move_curve1, smooth_move_curve_2)
            # Get the longitude and latitude of the city
            lng, lat = cityId_loc_dict[i]
            # Prepare city information: city name, longitude, latitude
            info = f"{cityId_city_dict[i]},{lng},{lat}"
            # Write the information and calculated values to the file
            fw.write(f"{info},{pearson_corr},{p_value}\n")



# Number of cities
city_num = 160

# Read attention flow and migration flow files to create matrices for both flows
attention_array, move_array = read_attention_move_file(city_num)

# Create dictionaries with information about the cities
# cityId_loc_dict: dictionary mapping city IDs to their locations (longitude and latitude)
# cityId_city_dict: dictionary mapping city IDs to their names
# city_cityId_dict: dictionary mapping city names to their IDs
cityId_loc_dict, cityId_city_dict, city_cityId_dict = tools.get_city_information()

# Get a list of all coordinates from the city location dictionary
all_loc_list = cityId_loc_dict.values()



# Utilizing migration flow data, measure the shape similarity of structural patterns between the outflow network and the inflow network for a node.
move_in_out_pearson_sim_fun()

# Utilizing attention flow data, measure the shape similarity of structural patterns between the outflow network and the inflow network for a node.
attention_in_out_pearson_sim_fun()

# Utilizing migration flow and attention flow data, measure the shape similarity of structural patterns between the attention flow network and the migration flow network for a node.
attention_move_pearson_sim_fun()






