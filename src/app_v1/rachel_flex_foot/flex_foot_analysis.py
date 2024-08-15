import sys
sys.path.append("C:/Users/tealw/Documents/PlatformIO/Projects/SmartHammer-1/src/app_v1")
from cuff_hammer_app_v1 import get_reshaped_array_from_arduino_csv
from cuff_hammer_app_v1 import plot_heat_map
import numpy as np
import matplotlib.pyplot as plt

DATA_LENGTH = 200

rachel_no_sig_gen_files = [
    [['src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/cuff_rachel_active_1.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/cuff_rachel_active_2.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/cuff_rachel_active_3.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/cuff_rachel_active_4.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/cuff_rachel_active_5.csv'],
    ['src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/hammer_rachel_active_1.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/hammer_rachel_active_2.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/hammer_rachel_active_3.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/hammer_rachel_active_4.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/hammer_rachel_active_5.csv']],
    
    [['src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/cuff_rachel_passive_1.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/cuff_rachel_passive_2.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/cuff_rachel_passive_3.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/cuff_rachel_passive_4.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/cuff_rachel_passive_5.csv'],
    ['src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/hammer_rachel_passive_1.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/hammer_rachel_passive_2.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/hammer_rachel_passive_3.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/hammer_rachel_passive_4.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/hammer_rachel_passive_5.csv']],

    [['src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/cuff_rachel_reflex_1.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/cuff_rachel_reflex_2.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/cuff_rachel_reflex_3.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/cuff_rachel_reflex_4.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/cuff_rachel_reflex_5.csv'],
    ['src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/hammer_rachel_reflex_1.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/hammer_rachel_reflex_2.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/hammer_rachel_reflex_3.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/hammer_rachel_reflex_4.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_no_sig_gen/hammer_rachel_reflex_5.csv']],
]

rachel_sig_gen_files = [
    [['src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/cuff_rachel_sig_gen_active_1.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/cuff_rachel_sig_gen_active_2.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/cuff_rachel_sig_gen_active_3.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/cuff_rachel_sig_gen_active_4.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/cuff_rachel_sig_gen_active_5.csv'],
    ['src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/hammer_rachel_sig_gen_active_1.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/hammer_rachel_sig_gen_active_2.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/hammer_rachel_sig_gen_active_3.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/hammer_rachel_sig_gen_active_4.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/hammer_rachel_sig_gen_active_5.csv']],
    
    [['src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/cuff_rachel_sig_gen_passive_1.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/cuff_rachel_sig_gen_passive_2.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/cuff_rachel_sig_gen_passive_3.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/cuff_rachel_sig_gen_passive_4.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/cuff_rachel_sig_gen_passive_5.csv'],
    ['src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/hammer_rachel_sig_gen_passive_1.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/hammer_rachel_sig_gen_passive_2.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/hammer_rachel_sig_gen_passive_3.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/hammer_rachel_sig_gen_passive_4.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/hammer_rachel_sig_gen_passive_5.csv']],

    [['src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/cuff_rachel_sig_gen_reflex_1.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/cuff_rachel_sig_gen_reflex_2.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/cuff_rachel_sig_gen_reflex_3.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/cuff_rachel_sig_gen_reflex_4.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/cuff_rachel_sig_gen_reflex_5.csv'],
    ['src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/hammer_rachel_sig_gen_reflex_1.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/hammer_rachel_sig_gen_reflex_2.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/hammer_rachel_sig_gen_reflex_3.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/hammer_rachel_sig_gen_reflex_4.csv',
     'src/app_v1/rachel_flex_foot/rachel_flex_foot_sig_gen/hammer_rachel_sig_gen_reflex_5.csv']],
]

chosen_folder = rachel_no_sig_gen_files
active_trials, passive_trials, reflex_trials = chosen_folder[0], chosen_folder[1], chosen_folder[2]

for i in range(len(active_trials[0])):
    active_data = get_reshaped_array_from_arduino_csv([active_trials[1][i], active_trials[0][i]], DATA_LENGTH)
    passive_data = get_reshaped_array_from_arduino_csv([passive_trials[1][i], passive_trials[0][i]], DATA_LENGTH)
    reflex_data = get_reshaped_array_from_arduino_csv([reflex_trials[1][i], reflex_trials[0][i]], DATA_LENGTH)

    adj_reflex = np.array(reflex_data[4]) - np.array(passive_data[4])
    new_cuff_data = [reflex_data[0], reflex_data[1], reflex_data[2], reflex_data[3], adj_reflex, reflex_data[5], reflex_data[6]]

    pulse_num = 15
    times = reflex_data[3][pulse_num]
    voltages = reflex_data[4][pulse_num]
    voltages2 = reflex_data[4][0]
    plt.plot(times, voltages)
    plt.plot(times, voltages2)
    print(f"Trial {i+1}")
    plt.show()

    #print("Analyzing: ")
    #plot_heat_map(new_cuff_data, folder_path = chosen_folder[0][0][0][:chosen_folder[0][0][0].rindex("/")]+"/", png_name="trial "+str(i+1), normalize_to_initial=False) 
