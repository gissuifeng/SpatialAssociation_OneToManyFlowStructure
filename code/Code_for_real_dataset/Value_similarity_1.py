"""
Interactions within the same network formed in two different but closely spaced time periods, primarily testing the value similarity of sequences.
"""

import numpy as np
import tools
import matplotlib.pyplot as plt
import matplotlib


def read_move_two_month_DTW_sim_fun(city_cityId_dict):
    cities = city_cityId_dict.keys()
    # Read data and save the data for two months separately into dictionaries
    jun_dict, jul_dict = {}, {}
    with open("../../dataset/real_dataset/migration_flow_data_June_July.csv", 'r', encoding='utf-8') as fr:
        fr.readline()  # Skip the header row
        for line in fr:
            attrs = line.strip().split(",")
            # Both cities must be in the dictionary, and there must be interaction values
            if attrs[1] in cities and attrs[2] in cities and len(attrs[3]) > 0 and len(attrs[4]) > 0:
                month_id, oid, did, v1, v2 = attrs[0], city_cityId_dict[attrs[1]], city_cityId_dict[attrs[2]], round(
                    float(attrs[3]), 3), round(float(attrs[4]), 3)
                if "Jun" in month_id:  # If the month is June
                    jun_dict[str(oid) + "," + str(did)] = (v1, v2)
                elif "Jul" in month_id:  # If the month is July
                    jul_dict[str(oid) + "," + str(did)] = (v1, v2)

    jun_move_array, jul_move_array = np.zeros((city_num, city_num)), np.zeros((city_num, city_num))
    for key, (v1, v2) in jun_dict.items():
        oid_str, did_str = key.split(",")
        v3, v4 = jun_dict.get(str(did_str) + "," + str(oid_str), (0, 0))
        jun_move_array[int(oid_str), int(did_str)] = round(v2 * v3, 5)

    for key, (v1, v2) in jul_dict.items():
        oid_str, did_str = key.split(",")
        v3, v4 = jul_dict.get(str(did_str) + "," + str(oid_str), (0, 0))
        jul_move_array[int(oid_str), int(did_str)] = round(v2 * v3, 5)

    return jun_move_array, jul_move_array


# Read attention flow and migration flow files to create matrices for attention flow and migration flow
# city_num: Number of cities, used to control the size of the matrices
def read_attention_move_file(city_num):
    # --------------------- Read attention flow file and create flow matrix -----------------------
    attention_array = np.zeros((city_num, city_num))
    with open("../../dataset/real_dataset/attention_flow.csv") as fr:  # Attention flow
        fr.readline()  # Skip the header row
        for line in fr:
            attrs = line.strip().split(",")
            start_city, end_city, flow_size = int(attrs[0]), int(attrs[1]), round(float(attrs[2]), 2)
            attention_array[start_city, end_city] = flow_size

    # --------------------- Read migration flow file and create flow matrix -----------------------
    move_array = np.zeros((city_num, city_num))
    with open("../../dataset/real_dataset/migration_flow.csv") as fr:  # Migration flow
        fr.readline()  # Skip the header row
        for line in fr:
            attrs = line.strip().split(",")
            start_city, end_city, flow_size = int(attrs[0]), int(attrs[1]), round(float(attrs[2]), 2)
            move_array[start_city, end_city] = flow_size

    return attention_array, move_array



# Compare the value similarity of migration flow strength between June and July
def move_jun_jul_strength_sim_fun(jun_move_array, jul_move_array):
    with open("../../result/value_similarity_results/Migration_flow_June_July_strength_value_similarity.csv", 'w') as fw:
        fw.write("Similar cities, longitude, latitude, DTW distance (strength), p-value\n")
        for i in range(city_num):
            # Smooth the curves for June and July migration flow using Savitzky-Golay filter
            smooth_attention_curve1 = tools.sg_smooth(jun_move_array[i])
            smooth_attention_curve_2 = tools.sg_smooth(jul_move_array[i])
            # Plot the smoothed curves for visual comparison
            tools.plot_curve(smooth_attention_curve1, smooth_attention_curve_2, title="migration_flow, node " + str(i) + ",")
            # Calculate the DTW distance and p-value between the smoothed curves
            DTW_distance, p_value = tools.curve_dtw_similarity_p_test(smooth_attention_curve1, smooth_attention_curve_2)
            # Get the longitude and latitude of the city
            lng, lat = cityId_loc_dict[i]
            # Prepare city information: city name, longitude, latitude
            info = str(cityId_city_dict[i]) + "," + str(lng) + "," + str(lat)
            # Write the information and calculated values to the file
            fw.write(info + "," + str(DTW_distance) + "," + str(p_value) + "\n")

# Write the flow strength data for June and July to separate files
def write_move_two_month_strength_fun(jun_move_array, cityId_loc_dict, cityId_city_dict):
    with open("../../dataset/real_dataset/migration_flow_strength_June.csv", 'w') as fw:
        fw.write("ocity,ox,oy,dcity,dx,dy,strength(%%)\n")
        for i in range(len(jun_move_array)):
            for j in range(len(jun_move_array[i])):
                ocity, dcity = cityId_city_dict[i], cityId_city_dict[j]
                oloc, dloc = cityId_loc_dict[i], cityId_loc_dict[j]
                fw.write(f"{ocity},{oloc[0]},{oloc[1]},{dcity},{dloc[0]},{dloc[1]},{jun_move_array[i,j]}\n")

    with open("../../dataset/real_dataset/migration_flow_strength_July.csv", 'w') as fw2:
        fw2.write("ocity,ox,oy,dcity,dx,dy,strength(%%)\n")
        for i in range(len(jul_move_array)):
            for j in range(len(jul_move_array[i])):
                ocity, dcity = cityId_city_dict[i], cityId_city_dict[j]
                oloc, dloc = cityId_loc_dict[i], cityId_loc_dict[j]
                fw2.write(f"{ocity},{oloc[0]},{oloc[1]},{dcity},{dloc[0]},{dloc[1]},{jul_move_array[i,j]}\n")

# Number of cities
city_num = 160

# Read attention flow and migration flow files to create matrices for attention flow and migration flow
attention_array, move_array = read_attention_move_file(city_num)
# Create dictionaries containing information about cities
cityId_loc_dict, cityId_city_dict, city_cityId_dict = tools.get_city_information()
# Get the coordinates of all points
all_loc_list = cityId_loc_dict.values()

# Read attention flow and migration flow files to create matrices for attention flow and migration flow
attention_array, move_array = read_attention_move_file(city_num)

# Read migration flow data for June and July to compare their value similarities
jun_move_array, jul_move_array = read_move_two_month_DTW_sim_fun(city_cityId_dict)
# Write the flow strength data for June and July to files
write_move_two_month_strength_fun(jun_move_array, cityId_loc_dict, cityId_city_dict)

# Compare the value similarity of migration flow strength between June and July
move_jun_jul_strength_sim_fun(jun_move_array, jul_move_array)





