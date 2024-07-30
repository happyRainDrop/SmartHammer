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

# Stuff for user to edit.
ports = ['COM9', 'COM20']  # hammer port, cuff port
baud_rate = 115200
DATA_LENGTH =  200 # From cuff arduino
files_folder_path = 'src/app_v1/'

##################################################################################################################################

def is_valid_data(line, port):
    """
    Check if the line matches the format (positive or negative floating point number), (positive or negative floating point number)
    """
    cuff_pattern = re.compile(r'^-?\d+(\.\d+)?, -?\d+(\.\d+)?$')
    hammer_pattern = re.compile(r'^-?\d+(\.\d+)?, -?\d+(\.\d+)?, -?\d+(\.\d+)?$')

    if (port == ports[0]): return hammer_pattern.match(line) is not None
    if (port == ports[1]): return cuff_pattern.match(line) is not None

def read_from_serial(port, baud_rate, output_file, done_event):
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

def plot_heat_map(output_files, DATA_LENGTH, png_name = "cuff_hammer_emg_combined"):
    hammer_csv = pd.read_csv(output_files[0], header=None).to_numpy()
    cuff_csv = pd.read_csv(output_files[1], header=None).to_numpy()

    hammer_times = hammer_csv[:,0]
    hammer_recieved = hammer_csv[:,1]
    emg_recieved = hammer_csv[:,2]
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
    
    # Plot using GridSpec
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.05])

    # Hammer and EMG signal subplot
    ax1 = plt.subplot(gs[0])
    ax1.plot(hammer_times, hammer_recieved, color="blue", label="Hammer strike")
    ax1.plot(hammer_times, emg_recieved, color="red", label="EMG signal")
    ax1.set_xlim(time_ticks[0], time_ticks[-1])
    ax1.set_title('Hammer and EMG voltage vs time')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_xlabel('Time (ms)')
    ax1.legend()

    # Cuff signal subplot
    ax2 = plt.subplot(gs[1])
    cuff_vals_for_heatmap = np.asarray(cuff_recieved_reshaped) - np.asarray(cuff_recieved_reshaped)[0, :]
    lower_outliers, upper_outliers, lower_lim_imshow, upper_lim_imshow = find_outliers_std(cuff_vals_for_heatmap)
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

    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()


    # Save the figure before showing it
    fig.savefig(files_folder_path + png_name + '.png')
    plt.show()
    plt.close(fig)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

    parser.add_argument('filename_suffix', type=str, help = 'appended to name of hammer,cuff csvs (do not include .csv postfix)') 
    parser.add_argument('--file_path', type=str, help='to change the default path (can insert full or relative path)', nargs='?') 

    args = parser.parse_args()
    if args.file_path is not None: files_folder_path = args.file_path

    output_files = [files_folder_path+'hammer_'+str(args.filename_suffix)+'.csv', 
                    files_folder_path+'cuff_'+str(args.filename_suffix)+'.csv']

    
    #'''
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
    # '''

    # Proceed to the next step of analyzing both CSVs
    print("Both threads are done. Proceeding to analyze the CSV files.")

    test_output_files = ["src/logs/exp_6_no_remove_cuff/Arduino_data/hammer_trial_1_rachel.txt", 
                         "src/logs/exp_6_no_remove_cuff/Arduino_data/cuff_trial_1_rachel.txt"]
    plot_heat_map(output_files, DATA_LENGTH, str(args.filename_suffix))
