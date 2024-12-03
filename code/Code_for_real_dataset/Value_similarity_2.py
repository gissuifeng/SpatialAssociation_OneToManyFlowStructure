"""
For the same spatial interaction network, the one-to-many structural similarity of inflow and outflow formed by migration flow/attention flow respectively
"""


import numpy as np
import tools
import matplotlib.pyplot as plt
import matplotlib

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


# Attention flow inflow, outflow, value similarity
def attention_in_out_DTW_sim_fun():
    attention_array_T = attention_array.T
    fw = open("../../result/value_similarity_results/attention_inflow_outflow_value similarity_result.csv", 'w')
    fw.write("cities, longitude, latitude, DTW distance, p-value\n")
    for i in range(city_num):
        smooth_attention_curve1, smooth_attention_curve_2 = tools.sg_smooth(attention_array_T[i]), tools.sg_smooth(attention_array[i])
        tools.plot_curve(smooth_attention_curve1, smooth_attention_curve_2, title="attentioin flow，node "+str(i)+",")
        DTW_distance, p_value = tools.curve_dtw_similarity_p_test(smooth_attention_curve1, smooth_attention_curve_2)
        (lng, lat) = cityId_loc_dict[i]
        info = str(cityId_city_dict[i]) + "," + str(lng) + "," + str(lat)  # cities, longitude, latitude
        fw.write(info + "," + str(DTW_distance) + "," + str(p_value) + "\n")

    fw.close()

# Migration flow inflow, outflow, value similarity
def move_in_out_DTW_sim_fun():
    move_array_T = move_array.T
    fw = open("../../result/value_similarity_results/move_inflow_outflow_value similarity_result.csv", 'w')
    fw.write("cities, longitude, latitude, DTW distance, p-value\n")
    for i in range(city_num):
        smooth_move_curve1, smooth_move_curve_2 = tools.sg_smooth(move_array_T[i]), tools.sg_smooth(move_array[i])
        tools.plot_curve(smooth_move_curve1, smooth_move_curve_2, title="move flow，node "+str(i)+",")
        DTW_distance, p_value = tools.curve_dtw_similarity_p_test(smooth_move_curve1, smooth_move_curve_2)
        (lng, lat) = cityId_loc_dict[i]
        info = str(cityId_city_dict[i]) + "," + str(lng) + "," + str(lat)  # cities, longitude, latitude
        fw.write(info + "," + str(DTW_distance) + "," + str(p_value) + "\n")

    fw.close()



# 城市数量
city_num = 160

# 读取关注流，迁徙流文件创建关注流的矩阵 和 迁徙流的矩阵
attention_array,  move_array  = read_attention_move_file(city_num)
# 创建有关城市信息的字典
cityId_loc_dict, cityId_city_dict, city_cityId_dict = tools.get_city_information()
# 获取所有点的坐标集合
all_loc_list = cityId_loc_dict.values()

# Utilizing migration flow data, measure the value similarity of structural patterns between the outflow network and the inflow network for a node.
move_in_out_DTW_sim_fun()

# Utilizing attentioin flow data, measure the value similarity of structural patterns between the outflow network and the inflow network for a node.
attention_in_out_DTW_sim_fun()










