'''
Name: mvp_analyze.py
Last updated: 7/11/25 by Ruth Berkun

Table of contents:
    Functions to parse Arduino Wifi data:
        get_wifi_data():
            Reads in data from Arduino and saves it to a CSV
    Functions to analyze cuff data:
        get_reshaped_array_from_arduino_csv(output_files, DATA_LENGTH, use_emg = False):
            Reads in Arduino cuff and hammer csvs, to output data in format needed for plot_heat_map
        plot_heat_map(input_files, folder_path = files_folder_path, png_name = "cuff_hammer_emg_combined", stddev = 3, use_emg = False):
            Plots hammer hit versus cuff heatmap, and allows user to select an area to search for the maximum intensity in.


Instructions for use: 
    RUNNING A LIVE EXPERIMENT: 
        In testing mode:
            Connect to Wifi (currently Ruth's hotspot, "Forest fire")
            Set read_live_data to true
                
            Run python mvp_analyze.py in terminal.
            1. Do NOT hit the hammer until the terminal prints
                Starting TCP server on port 4210...
                Waiting for connection from Arduino...

                It shoulnd't print anything after that if you're not hitting the hammer.
                If it prints something, close the terminal and restart the program.

            2. Hit the hammer on the table. (If triggered, checklight will turn on)
                It should now print something like
                Connected by ('192.168.235.110', 55104)
                (May take a while -- Arduino needs to connect to WiFi first)

                If it's not printing anything, the connection is  didn't trigger, try hitting it harder.

            3. Wait until prints "Proceeding to analyze the CSV files."
            On the picture that pops up, click and drag a rectangle of the region you want to find the max
            intensity in.
            Close the image, and the program will complete, and save to the location you specified

            4. Example of what is printed to terminal on a successful run:

        To specify folder path and or file name to save csv and png files under: 
            Run python cuff_hammer_app_v1.py --filename_suffix DESIRED_NAME --folder_path DESIRED_FOLDER_PATH

    ANALYZING PREVIOUS EXPERIMENT DATA:
        Set read_live_data to false
        Run python cuff_hammer_app_v1.py --filename_suffix DESIRED_NAME --folder_path
        For example, if you want to analyze folder1/hammer_test.csv and folder1/cuff_test.csv, you would run
            python cuff_hammer_app_v1.py --filename_suffix test --folder_path folder1
'''

import socket

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import time

import argparse
import matplotlib.gridspec as gridspec

from matplotlib.widgets import RectangleSelector

# Stuff for user to edit
DATA_LENGTH =  260 # From cuff arduino
read_live_data = True
demo_mode = False       # Use it to pretend to generate a heat map on the spot, but
                        # it actually just pulls up Sina's old reflex :p

# Default folder path and file name used in testing mode
files_folder_path = 'logs/'
file_name = 'test'

##################################################################################################################################

def read_from_TCP(filename):
    """
    Through a TCP connection to the Arduino Giga, saves data transmitted to CSV
    Inputs:
        filename -- path to save to
    Ouputs:
        CSV at specified file and folder path
    """
    HOST = '0.0.0.0'
    PORT = 4210
    LINES_PER_PACKET = 100
    CSV_FILENAME = filename
    current_packet_id = 0
    last_acked_id = -1
    start_time = 0

    print(f"Starting TCP server on port {PORT}...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print("Waiting for connection from Arduino...")

        conn, addr = server_socket.accept()
        start_time = time.time()
        print(f"Connected by {addr}")

        buffer = b""
        linesToWrite = []

        packet_start_time = time.time()
        csv_start_time = time.time()

        with open(CSV_FILENAME, 'w') as f:
            while True:
                data = conn.recv(65536)
                if not data:
                    print("Connection closed.")
                    break

                buffer += data
                lines = buffer.split(b'\n')
                buffer = lines[-1]  # Keep last partial line

                for line in lines[:-1]:
                    try:
                        line_str = line.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        continue

                    if "====" in line_str:
                        print("✅ End-of-stream received. Exiting.")
                        conn.close()
                        start_time = time.time() - start_time
                        print(f"Elapsed time for TCP transmit: {start_time} seconds")
                        return

                    if line_str.startswith("PACKET"):
                        try:
                            current_packet_id = int(line_str[6:].strip())
                            lines_in_packet = 0
                            #packet_start_time = time.time()
                            # print(f"📦 PACKET {current_packet_id}")
                        except ValueError:
                            continue

                    elif line_str.count(',') == 5 and current_packet_id is not None:
                        lines_in_packet += 1
                        linesToWrite.append(line_str)
                        if lines_in_packet >= LINES_PER_PACKET and current_packet_id != last_acked_id:
                            
                            #packet_start_time = time.time() - packet_start_time
                            #print(f"  Read in 1 packet: {packet_start_time*1000} ms")

                            # Batch write to CSV
                            #packet_start_time = time.time()
                            for i in range(LINES_PER_PACKET):
                                f.write(linesToWrite[i] + '\n')
                            linesToWrite = []
                            packet_start_time = time.time() - packet_start_time
                            #print(f"  CSV write in 1 packet: {packet_start_time*1000} ms")

                            # Send ACK
                            ack_msg = f"ACK{current_packet_id}\n".encode('utf-8')
                            # print("\tACK"+str(current_packet_id))
                            conn.sendall(ack_msg)
                            last_acked_id = current_packet_id
                    
                    else:
                        print(f"!! Malformed line: {line_str}")


def read_from_UDP(filename):
    """
    Listens for UDP packets from an Arduino and saves received data to a CSV file
    when a termination line containing '==' is received.
    
    Parameters:
        filename (str): Filename for the output CSV file.
    """
    import socket

    PORT = 4210
    LINES_PER_PACKET = 10
    CSV_FILENAME = filename

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', PORT))
    sock.settimeout(300.0) # 5-minute timeout before stop listening
    print("Listening for UDP packets...")

    with open(CSV_FILENAME, 'w') as f:
        current_packet = None
        lines_in_packet = 0
        last_acked = -1
        buffer = b""

        while True:
            data, addr = sock.recvfrom(65536)
            
            buffer += data
            lines = buffer.split(b'\n')
            buffer = lines[-1]  # Keep last partial line

            for line in lines:
                try:
                    line_str = line.decode('utf-8').strip()
                except UnicodeDecodeError:
                    continue

                # print(line_str)

                if "====" in line_str:
                    print("✅ End-of-stream received.")
                    exit()

                elif line_str.startswith("PACKET"):
                    try:
                        current_packet = int(line_str[6:].strip())
                        lines_in_packet = 0
                        # print(f"📦 PACKET {current_packet}")
                    except ValueError:
                        # print(f"⚠️ Malformed PACKET line: {line_str}")
                        continue

                elif line_str.count(',') == 5 and current_packet is not None:
                    f.write(line_str + '\n')
                    lines_in_packet += 1
                    if lines_in_packet >= LINES_PER_PACKET and current_packet != last_acked:
                        ack = f"ACK{current_packet}"
                        sock.sendto(ack.encode(), addr)
                        last_acked = current_packet

                elif len(line_str) > 0:
                    print(f"!! Malformed line: {line_str}")
                
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

def get_reshaped_array_from_arduino_csv(output_file, DATA_LENGTH, use_emg = False):
    '''
    Reads in Arduino cuff and hammer csvs, to output data in format needed for plot_heat_map \n
    Inputs:
        output_file: path to csv file to analyze
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
    my_csv = pd.read_csv(output_file, header=None).to_numpy()

    cuff_times = my_csv[:,0]
    cuff_recieved = my_csv[:,2]
    hammer_recieved = my_csv[:,4]
    hammer_times = cuff_times
    emg_recieved = my_csv[:,5]
    
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

def plot_heat_map(input_files, folder_path = files_folder_path, png_name = "cuff_hammer_emg_combined", stddev = 3, use_emg = False, normalize_to_initial = True):
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
    start_index = 5
    hammer_times = input_files[0][start_index:]
    hammer_recieved = input_files[1][start_index:]
    emg_recieved = input_files[2][start_index:]
    cuff_times_reshaped = input_files[3][start_index:]
    cuff_recieved_reshaped = input_files[4][start_index:]
    time_ticks = input_files[5][start_index:]
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
    cuff_vals_for_heatmap = np.asarray(cuff_recieved_reshaped)
    if (normalize_to_initial): 
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
    parser.add_argument('--folder_path', type=str, help='to change the default path (can insert full or relative path)', nargs='?') 

    args = parser.parse_args()
    if args.folder_path is not None: files_folder_path = args.folder_path
    if args.filename_suffix is not None: file_name = str(args.filename_suffix)

    output_file = files_folder_path+'Arduino_'+file_name+'.csv'

    if (read_live_data): read_from_TCP(output_file)        
    if (demo_mode): output_file = "sample_data.csv"

    '''
    my_csv = pd.read_csv(output_file, header=None).to_numpy()
    cuff_times = my_csv[:,0]
    cuff_recieved = my_csv[:,1]
    hammer_recieved = my_csv[:,4]
    plt.plot(cuff_times, cuff_recieved)
    plt.xlim(0, 600)
    plt.show()
    #'''
    
    #'''
    data_arrays = get_reshaped_array_from_arduino_csv("logs/Arduino_test.csv", DATA_LENGTH)
    print(f"Reading {output_file}")
    plot_heat_map(data_arrays, png_name=file_name, folder_path=files_folder_path)
    #s'''
