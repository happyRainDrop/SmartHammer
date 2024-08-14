'''
Name: cuff_hammer_app_v1.py
Last updated: 8/14/26 by Ruth Berkun

Table of contents:
    Functions to parse Arduino serial data:
        read_serial_data(port, baud_rate, output_file, done_event):
            Reads in data from specific Arduino and saves it to a CSV
        is_valid_data(line, port):
            Is this line valid coming from the specified Arduino?
    Functions to analyze cuff data:
        get_reshaped_array_from_arduino_csv(output_files, DATA_LENGTH, use_emg = False):
            Reads in Arduino cuff and hammer csvs, to output data in format needed for plot_heat_map
        plot_heat_map(input_files, folder_path = files_folder_path, png_name = "cuff_hammer_emg_combined", stddev = 3, use_emg = False):
            Plots hammer hit versus cuff heatmap, and allows user to select an area to search for the maximum intensity in.


Instructions for use: 
    RUNNING A LIVE EXPERIMENT: 
        In testing mode:
            Set read_live_data to true
            Change ports to be the name of the COM ports used for your hammer and cuff Arduinos
                (varies laptop to laptop)
            baud_rate and DATA_LENGTH are hard-coded in the cuff and hammer Arduinos. 
            Currently 115200 and 200 respectively.
                
            Run python cuff_hammer_app_v1.py in terminal.
        To specify folder path and or file name to save csv and png files under: 
            Run python cuff_hammer_app_v1.py --filename_suffix DESIRED_NAME --folder_path DESIRED_FOLDER_PATH

    ANALYZING PREVIOUS EXPERIMENT DATA:
        Set read_live_data to false
        Run python cuff_hammer_app_v1.py --filename_suffix DESIRED_NAME --folder_path
        For example, if you want to analyze folder1/hammer_test.csv and folder1/cuff_test.csv, you would run
            python cuff_hammer_app_v1.py --filename_suffix test --folder_path folder1
'''

import serial
import time
import csv
import re
import threading

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

import argparse
import matplotlib.gridspec as gridspec

from matplotlib.widgets import RectangleSelector

# Stuff for user to edit.
ports = ['COM9', 'COM20']  # hammer port, cuff port
baud_rate = 115200
DATA_LENGTH =  200 # From cuff arduino
read_live_data = False

# Default folder path and file name used in testing mode
files_folder_path = 'src/app_v1/'
file_name = 'test'

##################################################################################################################################

def is_valid_data(line, port):
    """
    Check if the line matches the format (positive or negative floating point number), (positive or negative floating point number) \n
    Inputs:
        line: Line to check if it's valid serial data we want to save.
        port: 'COM9' for example
    Ouputs:
        true if valid, false if not
    """
    cuff_pattern = re.compile(r'^-?\d+(\.\d+)?, -?\d+(\.\d+)?$')
    hammer_pattern = re.compile(r'^-?\d+(\.\d+)?, -?\d+(\.\d+)?, -?\d+(\.\d+)?$')

    if (port == ports[0]): return hammer_pattern.match(line) is not None
    if (port == ports[1]): return cuff_pattern.match(line) is not None

def read_from_serial(port, baud_rate, output_file, done_event):
    '''
    Reads in data from specific Arduino and saves it to a CSV. Starts saving data once
    it reads in a valid line (as specified by helped function is_valid data). Stops saving
    data once it reads in an invalid line after saving >0 valid lines.

    Inputs:
        port: 'COM9' for example
        baud_rate: Serial baud rate specified on Arduino program
        output_file: Path to save csv to
        done_event: Threading event, used to tell us when we are done reading Serial
    Outputs:
        csv saved to path specified at output_file
    '''
    try:
        # Initialize the serial connection
        ser = serial.Serial(port, baud_rate)
        time.sleep(2)  # Wait for the connection to be established

        print(f"Connected to {port} at {baud_rate} baud rate.")
        
        # Open the CSV file for writing
        with open(output_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            saving_data = False
            done_writing_data = False

            # Continuously read from the serial port
            while not done_writing_data:

                if ser.in_waiting > 0:
                    line = ser.readline()
                    try:
                        line = line.decode('utf-8').rstrip()                    
                        # If valid data, save to csv
                        if is_valid_data(line, port):
                            if not saving_data:
                                print(f"{port}: Starting to save data to CSV, found line {line} is valid")
                            saving_data = True
                            csvwriter.writerow(line.split(','))
                        # Otherwise, we are done reading.
                        else:
                            if saving_data:
                                print(f"{port}: Stopping data saving to CSV")
                                done_writing_data = True
                            saving_data = False

                    except:
                        print(f"{port}: line {line} invalid.")
                        saving_data = False
                        
    except serial.SerialException as e:
        print(f"Error on {port}: {e}")
    except KeyboardInterrupt:
        print(f"Exiting program on {port}.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print(f"Serial connection closed on {port}.")
        done_event.set()

##################################################################################################################################

def find_outliers_std(data, threshold=3):
    '''
    Helper function to set minimum and maximum bounds on data that exclude outliers \n
    Inputs:
        data: 1D array to analyze
        threshhold: Data outside [threshold] standard deviations will not be included within the minimum, maximum bounds
    Outputs:
        lower_outliers: Values of data below [threshold] standard deviations from the mean
        upper_outliers: Values of data above [threshold] standard deviations from the mean
        lower_bound: Number exactly at [threshold] standard deviations below the mean
        uppder_bound: Number exactly at [threshold] standard deviations above the mean
    '''

    # Calculate the mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)
    
    # Determine the lower and upper bounds
    lower_bound = mean - threshold * std_dev
    upper_bound = mean + threshold * std_dev
    
    # Find outliers
    lower_outliers = data[data < lower_bound]
    upper_outliers = data[data > upper_bound]
    
    return lower_outliers, upper_outliers, lower_bound, upper_bound

def get_reshaped_array_from_arduino_csv(output_files, DATA_LENGTH, use_emg = False):
    '''
    Reads in Arduino cuff and hammer csvs, to output data in format needed for plot_heat_map \n
    Inputs:
        output_files: array of the format [path to hammer csv, path to cuff csv]
        DATA_LENGTH: How many data points saved per recieved pulse in the cuff Arduino
        use_emg: Consider emg data (third col of hammer csv)
    Outputs:
        [hammer_times, hammer_recieved, emg_recieved, cuff_times_reshaped, cuff_recieved_reshaped, time_ticks, NUM_PULSES]
        hammer_times: Hammer times in ms. First col of hammer csv
        hammer_recieved: Hammer Arduino recieved voltages (V). Second col of hammer csv
        emg_recieved: EMG data in voltages (V), empty array if use_emg = False
        cuff_times_reshaped:[[times of recieved pulse 1], [times of recieved pulse 2],...]
        cuff_recieved_reshaped: [[voltages of recieved pulse 1], [voltages of recieved pulse 2],...]
        time_ticks: Used on heat maps -- the starting time of each pulse.
        NUM_PULSES: Number of recieved pulses detected in the cuff data. 
    '''
    hammer_csv = pd.read_csv(output_files[0], header=None).to_numpy()
    cuff_csv = pd.read_csv(output_files[1], header=None).to_numpy()

    hammer_times = hammer_csv[:,0]
    hammer_recieved = hammer_csv[:,1]
    emg_recieved = hammer_csv[:,2] if use_emg else []
    cuff_times = cuff_csv[:,0]
    cuff_recieved = cuff_csv[:,1]
    
    NUM_PULSES = int(len(cuff_recieved) / DATA_LENGTH)
    print(f"{NUM_PULSES} recieved pulses found.")

    # Reshape by pulse (NUM_PULSES rows, DATA_LENGTH columns)
    cuff_times_reshaped = []
    cuff_recieved_reshaped = []
    time_ticks = []
    i = 0
    for r in range(NUM_PULSES):
        cuff_pulse_times = []
        cuff_pulse_data = []
        
        for c in range(DATA_LENGTH):
            cuff_pulse_times.append(cuff_times[i])
            cuff_pulse_data.append(cuff_recieved[i])
            i+=1
        
        cuff_times_reshaped.append(cuff_pulse_times)
        cuff_recieved_reshaped.append(cuff_pulse_data)
        time_ticks.append(round(cuff_pulse_times[0], 2))

    return [hammer_times, hammer_recieved, emg_recieved, cuff_times_reshaped, cuff_recieved_reshaped, time_ticks, NUM_PULSES]

def plot_heat_map(input_files, folder_path = files_folder_path, png_name = "cuff_hammer_emg_combined", stddev = 3, use_emg = False):
    '''
    Plots hammer hit versus cuff heatmap, and allows user to select an area to search for the maximum intensity in. \n

    Inputs: \n
        input_files: [hammer_times, hammer_recieved, emg_recieved, cuff_times_reshaped, cuff_recieved_reshaped, time_ticks, NUM_PULSES]
            hammer_times: Hammer times in ms. First col of hammer csv
            hammer_recieved: Hammer Arduino recieved voltages (V). Second col of hammer csv
            emg_recieved: EMG data in voltages (V), empty array if use_emg = False
            cuff_times_reshaped:[[times of recieved pulse 1], [times of recieved pulse 2],...]
            cuff_recieved_reshaped: [[voltages of recieved pulse 1], [voltages of recieved pulse 2],...]
            time_ticks: Used on heat maps -- the starting time of each pulse.
            NUM_PULSES: Number of recieved pulses detected in the cuff data.
        folder_path: location to save png in 
        png_name: name of png that is saved 
        stddev: Sets limit of color map. higher stddev = less outliers unconsidered
        use_emg: True to plot EMG data on top of hammer data, false otherwise

    Outputs: \n
        Saves heatmap to folder_path + png_name + '.png'
    '''
    
    # Retrieve the data we need for the heat map
    hammer_times = input_files[0]
    hammer_recieved = input_files[1]
    emg_recieved = input_files[2]
    cuff_times_reshaped = input_files[3]
    cuff_recieved_reshaped = input_files[4]
    time_ticks = input_files[5]
    NUM_PULSES = input_files[6]

    # Plot using GridSpec
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.05])

    # Hammer and EMG signal subplot
    ax1 = plt.subplot(gs[0])
    ax1.plot(hammer_times, hammer_recieved, color="blue", label="Hammer strike")
    if use_emg: ax1.plot(hammer_times, emg_recieved, color="red", label="EMG signal")
    ax1.set_xlim(time_ticks[0], time_ticks[-1])
    if use_emg: ax1.set_title('Hammer and EMG voltage vs time') 
    else: ax1.set_title("Hammer voltage vs time")
    ax1.set_ylabel('Voltage (V)')
    ax1.set_xlabel('Time (ms)')
    ax1.legend()

    # Cuff signal subplot
    ax2 = plt.subplot(gs[1])
    cuff_vals_for_heatmap = np.asarray(cuff_recieved_reshaped) - np.asarray(cuff_recieved_reshaped)[0, :]
    lower_outliers, upper_outliers, lower_lim_imshow, upper_lim_imshow = find_outliers_std(cuff_vals_for_heatmap, stddev)
    im = ax2.imshow(np.transpose(cuff_vals_for_heatmap), aspect='auto', cmap='jet', vmin=lower_lim_imshow, vmax=upper_lim_imshow)
    ax2.set_title('Circuit envelope: \nPulse height vs time normalize to start of pulse, all pulses overlayed')
    ax2.set_ylabel('Array index within pulse')
    ax2.set_xlabel('Start time of pulse (ms)')
    time_tick_positions = np.arange(0, NUM_PULSES, NUM_PULSES / len(time_ticks))
    ax2.set_xticks(ticks=time_tick_positions[0::5])
    ax2.set_xticklabels(labels=time_ticks[0::5])
    ax2.tick_params(axis='x', rotation=90)

    # Colorbar subplot
    cbar_ax = plt.subplot(gs[2])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

    ############################################################# Stuff to select area and find max point
    max_amplitude_text = ax2.text(0, 0, '', color='white', fontsize=12, ha='center')
    max_point_marker, = ax2.plot([], [], 'ro')

    def on_select(eclick, erelease):
        # Get the coordinates of the rectangle
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Define the rectangle area
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        data = np.transpose(cuff_vals_for_heatmap)
        absdata = np.abs(np.transpose(cuff_vals_for_heatmap))

        # Get the subarray of the selected area
        selected_area = absdata[y_min:y_max+1, x_min:x_max+1]
        
        # Find the indices of the maximum value within the selected area
        max_idx = np.unravel_index(np.argmax(selected_area), selected_area.shape)
        max_y, max_x = max_idx[0] + y_min, max_idx[1] + x_min
        max_value = data[max_y, max_x]
        time_of_max = np.transpose(cuff_times_reshaped)[max_y, max_x]
        
        # Update the annotation and marker
        max_amplitude_text.set_position((max_x, max_y))
        max_amplitude_text.set_text(f'{time_of_max}, {max_value:.2f}')
        max_point_marker.set_data(max_x, max_y)
        
        # Print the row and column index of the maximum point
        print(f'Manual select found: Maximum muscle contraction found at {time_of_max} ms after hammer hit')
        
        # Redraw the figure to update the annotation and marker
        fig.canvas.draw_idle()

    # Create the RectangleSelector
    rect_selector = RectangleSelector(ax2, on_select, useblit=True,
                                        button=[1], minspanx=5, minspany=5, spancoords='pixels',
                                        interactive=True)


    # Save the figure before showing it
    plt.subplots_adjust(hspace=1)
    plt.show()
    str_name = folder_path + png_name + '.png'
    fig.savefig(str_name)
    print(f"Saving to: {str_name}")
    plt.close(fig)
    
    # Return time of reflex
    min_reflex_time = 28 # in ms
    max_reflex_time = 50 # in ms
    min_reflex_index_within_pulse = 30
    reflex_time = 0
    max_amplitude_heat_map = 0
    for r in range(NUM_PULSES):
        for c in range(min_reflex_index_within_pulse, len(cuff_recieved_reshaped[r])):
            amplitude_diff = cuff_recieved_reshaped[r][c] - cuff_recieved_reshaped[0][c]
            if (cuff_times_reshaped[r][c] < min_reflex_time or cuff_times_reshaped[r][c] > max_reflex_time): continue
            if np.abs(amplitude_diff) > max_amplitude_heat_map and amplitude_diff > lower_lim_imshow and amplitude_diff < upper_lim_imshow:
                max_amplitude_heat_map = np.abs(amplitude_diff)
                reflex_time = cuff_times_reshaped[r][c]
    print(f"Auto-detect found: Maximum muscle contraction found at {reflex_time} ms after hammer hit.")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

    parser.add_argument('--filename_suffix', type=str, help = 'appended to name of hammer,cuff csvs (do not include .csv postfix)') 
    parser.add_argument('--file_path', type=str, help='to change the default path (can insert full or relative path)', nargs='?') 

    args = parser.parse_args()
    if args.file_path is not None: files_folder_path = args.file_path
    if args.filename_suffix is not None: file_name = str(args.filename_suffix)

    output_files = [files_folder_path+'hammer_'+file_name+'.csv', 
                    files_folder_path+'cuff_'+file_name+'.csv']

    
    if (read_live_data):
        # Threading to read both serial ports simultaneously.
        done_events = [threading.Event() for _ in ports]
        threads = []
        for port, output_file, done_event in zip(ports, output_files, done_events):
            thread = threading.Thread(target=read_from_serial, args=(port, baud_rate, output_file, done_event))
            threads.append(thread)
            thread.start()
        # Wait for both threads to complete
        for done_event in done_events:
            done_event.wait()

        # Proceed to the next step of analyzing both CSVs
        print("Both threads are done. Proceeding to analyze the CSV files.")

    test_output_files = ["src/app_v1/misc_trials/good_sina_trials/hammer_t1_sina_redo.csv", 
                         "src/app_v1/misc_trials/good_sina_trials/cuff_t1_sina_redo.csv"]
    
    data_arrays = get_reshaped_array_from_arduino_csv(output_files, DATA_LENGTH)
    print(f"Reading {output_files[0]} and {output_files[1]}")
    plot_heat_map(data_arrays, file_name)