import serial
import time
import csv
import re
import threading

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

##################################################################################################################################

def is_valid_data(line):
    """
    Check if the line matches the format (positive or negative floating point number), (positive or negative floating point number)
    """
    cuff_pattern = re.compile(r'^-?\d+(\.\d+)?, -?\d+(\.\d+)?$')
    hammer_pattern = re.compile(r'^-?\d+(\.\d+)?, -?\d+(\.\d+)?, -?\d+(\.\d+)?$')

    return cuff_pattern.match(line) is not None or hammer_pattern.match(line) is not None

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
                    line = ser.readline().decode('utf-8').rstrip()
                    
                    # If valid data, save to csv
                    if is_valid_data(line):
                        if not saving_data:
                            print(f"{port}: Starting to save data to CSV.")
                        saving_data = True
                        csvwriter.writerow(line.split(','))

                    # Otherwise, we are done reading.
                    else:
                        if saving_data:
                            print(f"{port}: Stopping data saving to CSV.")
                            done_writing_data = True
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

def plot_heat_map(output_files, DATA_LENGTH):
    hammer_csv = pd.read_csv(output_files[0]).to_numpy()
    cuff_csv = pd.read_csv(output_files[1]).to_numpy()

    hammer_times = hammer_csv[:,0]
    hammer_recieved = hammer_csv[:,1]
    emg_recieved = hammer_csv[:,1]
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
        time_ticks.append(cuff_pulse_times[0])
    
    # Plot
    # Hammer and EMG signal    
    plt.subplot(2,1,1)
    plt.plot(hammer_times, hammer_recieved, color = "blue", label = "Hammer strike")
    plt.plot(hammer_times, emg_recieved, color = "red", label = "EMG signal")
    plt.xlim(time_ticks[0], time_ticks[-1])

    plt.title('Hammer and EMG voltage vs time')
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (ms)')
    plt.legend()

    # Cuff signal
    plt.subplot(2,1,2)
    cuff_vals_for_heatmap = np.asarray(cuff_recieved_reshaped) - np.asarray(cuff_recieved_reshaped)[0,:]
    lower_outliers, upper_outliers, lower_lim_imshow, upper_lim_imshow = find_outliers_std(cuff_vals_for_heatmap)
    plt.imshow(np.transpose(cuff_vals_for_heatmap), aspect='auto', cmap='jet', vmin = lower_lim_imshow, vmax = upper_lim_imshow)

    plt.title('Circuit envelope: \nPulse height vs time normalize to start of pulse, all pulses overlayed')
    plt.ylabel('Array index within pulse')
    plt.xlabel('Start time of pulse (ms)')
    time_tick_positions = np.arange(0, NUM_PULSES, NUM_PULSES/len(time_ticks))
    plt.xticks(ticks = time_tick_positions[0::5], labels = time_ticks[0::5])
    plt.xticks(rotation=90)

    plt.subplots_adjust(hspace = 0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig("cuff_hammer_emg_combined.png")




if __name__ == "__main__":
    
    # Stuff for user to edit.
    ports = ['COM9', 'COM20']  # hammer port, cuff port
    baud_rate = 115200
    output_files = ['output_hammer.csv', 'output_cuff.csv']
    DATA_LENGTH =  200 # From cuff arduino

    '''
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
    plot_heat_map(test_output_files, DATA_LENGTH)
