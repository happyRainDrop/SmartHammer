'''
Name: mvp_analyze.py
Last updated: 7/18/25 by Ruth Berkun

Table of contents:
    Functions to parse Arduino Wifi data:
        write_samples(buf, writer):
            Helps read_from_TCP write in the floats in the buffer in the correct format for CSV
        read_from_TCP():
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
            Set read_live_data to true and run python mvp_analyze.py
            OR, just run python mvp_analyze.py --live_mode in terminal. 
            1. Wait for connection.
                📡 TCP server listening on 0.0.0.0:4210

            2. Hit the hammer on the table. (If triggered, checklight will turn on)
                It should now print something like
                ✅ Connected by ('10.60.225.147', 53533)
                Recieving packets... ━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   5% 0:01:51
                (May take a while -- Arduino needs to connect to WiFi first)

                If it's not printing anything, the connection is not working. Try power cycling Arduino.

            3. Wait until prints "Proceeding to analyze the CSV files."
            On the picture that pops up, click and drag a rectangle of the region you want to find the max
            intensity in.
            Close the image, and the program will complete, and save to the location you specified

            4. Example of what is printed to terminal on a successful run:
            📡 TCP server listening on 0.0.0.0:4210
            ✅ Connected by ('10.60.225.147', 53533)
            📝 Finished writing logs/Arduino_test.csv in 114.1 seconds.
            Recieving packets... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
            2000 recieved pulses found.
            Reading logs/Arduino_test.csv
            Saving to: logs/test.png

        To specify folder path and or file name to save csv and png files under: 
            Run python cuff_hammer_app_v1.py --filename_suffix DESIRED_NAME --folder_path DESIRED/FOLDER/PATH/
            It will save in "DESIRED/FOLDER/PATH/Arduino_DESIRED_NAME.csv".

    ANALYZING PREVIOUS EXPERIMENT DATA:
        Set read_live_data to false, or run with the --readback_mode suffix.
        Run python cuff_hammer_app_v1.py --filename_suffix DESIRED_NAME --folder_path
        For example, if you want to analyze folder1/hammer_test.csv and folder1/cuff_test.csv, you would run
            python cuff_hammer_app_v1.py --filename_suffix test --folder_path folder1
'''

import socket, struct, os, csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import re
import math

import argparse
import matplotlib.gridspec as gridspec
from rich.progress import Progress
from scipy.ndimage import uniform_filter1d

from matplotlib.widgets import RectangleSelector

# Stuff for user to edit
DATA_LENGTH =  260 # From cuff arduino: number of sensor readings per pulse
read_live_data = False
demo_mode = False       # Use it to pretend to generate a heat map on the spot, but
                        # it actually just pulls up Sina's old reflex :p

# Default folder path and file name used in testing mode
files_folder_path = 'logs/'
file_name = 'test'

# Other information hard-coded in the Arduino
HEADER_RE   = re.compile(br'^PACKET(\d+)\n')    # what do we send at the start of a packet?
NUM_PULSES_TO_SAVE = 2000
SAMPLES_PER_PACKET = 500
NUM_CHANNELS = 6                                # how many numbers in each line of the CSV?
TOTAL_PACKETS_EXPECTED = math.floor(NUM_PULSES_TO_SAVE * DATA_LENGTH / SAMPLES_PER_PACKET)
BYTES_PER_SAMPLE = 4 * NUM_CHANNELS
BYTES_PER_PACKET = BYTES_PER_SAMPLE * SAMPLES_PER_PACKET

##################################################################################################################################

def write_samples(buf, writer):
    """Write full samples from buf to CSV; each sample = 6 floats"""
    num_samples = len(buf) // BYTES_PER_SAMPLE
    for i in range(num_samples):
        offset = i * BYTES_PER_SAMPLE
        sample = struct.unpack('<6f', buf[offset:offset + BYTES_PER_SAMPLE])
        # Write with custom formatting: 6 decimals for time, 3 for others
        writer.writerow([f"{sample[0]:.6f}"] + [f"{v:.3f}" for v in sample[1:]])
    del buf[:num_samples * BYTES_PER_SAMPLE]  # consume written bytes

def read_from_TCP(filename):
    """
    Connects to Arduino through a TCP socket connection.
    Arduino sends samples of 4-byte floats over Wi-Fi, each sample has NUM_CHANNELS floats
    Arduino sends these samples in packets with SAMPLES_PER_PACKET samples each
    read_from_TCP processes these packets, tells the Arduino it recieved them, and sends them.
    Inputs:
        filename: full path of where to save the CSV.
    Ouputs:
        CSV of Arduino values (time in ms, cuff_1_voltage, cuff_2_voltage, cuff_3_voltage, 
            hammer_voltage, emg_voltage)
    """
    host, port = '0.0.0.0', 4210
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv, \
         open(filename, 'w', newline='') as fout:

        writer = csv.writer(fout)
        # writer.writerow(["time_sec", "cuff1_V", "cuff2_V", "cuff3_V", "hammer_V", "emg_V"])

        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port));  srv.listen(1)
        print(f"📡 TCP server listening on {host}:{port}")
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"✅ Connected by {addr}")
        

        with Progress() as progress:
            task = progress.add_task("[green]Recieving packets...", total=TOTAL_PACKETS_EXPECTED)

            start_time = time.time()
            data_bin_buf = bytearray()
            total_bin_buf = bytearray()
            last_acked_packet = -1

            try:
                while True:

                    # 1. Get data in  (raw bytes)
                    data = conn.recv(8192)
                    if not data:
                        write_samples(data_bin_buf, writer)  # flush remaining
                        break

                    total_bin_buf.extend(data)

                    # 2. Search for header and process bytes between headers. Give up 
                    # when we've found all headers
                    while (HEADER_RE.search(total_bin_buf) is not None):
                        header = HEADER_RE.search(total_bin_buf)

                        # Process: any bytes that came in BEFORE the header
                        hdr_start_index, hdr_end_index = header.span()
                        data_bin_buf.extend(total_bin_buf[:hdr_start_index])    # push bytes to data_bin_buf
                        pkt_no = int(header.group(1))
                        payload = data_bin_buf[:BYTES_PER_PACKET]

                        if len(payload) == BYTES_PER_PACKET:
                            if (last_acked_packet != pkt_no):
                                write_samples(payload, writer)
                                conn.sendall(f"ACK{pkt_no}\n".encode())
                                # print(f"✅ ACK{pkt_no}")
                                progress.update(task, advance=1)
                            else:
                                print("Warning: repeat packet {pkt_no}, ignoring.")
                            last_acked_packet = pkt_no
                            data_bin_buf = data_bin_buf[BYTES_PER_PACKET:]  # remove written part

                        # Remove header so next iteration we can look for bytes before next header
                        total_bin_buf = total_bin_buf[hdr_end_index:]  # remove those bytes from total_bin_buf

                    # 3. Write all bytes that came after the last header.
                    data_bin_buf.extend(total_bin_buf)
                    total_bin_buf = bytearray()
                    while (len(data_bin_buf) >= BYTES_PER_PACKET):
                        payload = data_bin_buf[:BYTES_PER_PACKET]
                        if len(payload) == BYTES_PER_PACKET:
                            if (last_acked_packet != pkt_no):
                                write_samples(payload, writer)
                                conn.sendall(f"ACK{pkt_no}\n".encode())
                                # print(f"✅ ACK{pkt_no}")
                                progress.update(task, advance=1)
                            else:
                                print("Warning: repeat packet {pkt_no}, ignoring.")
                            last_acked_packet = pkt_no
                            data_bin_buf = data_bin_buf[BYTES_PER_PACKET:]  # remove written part

    

            finally:
                duration = time.time() - start_time
                print(f"📝 Finished writing {filename} in {duration:.1f} seconds.")
                conn.close()
                
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
        [hammer_times, hammer_recieved, emg_recieved, cuff_times_reshaped, cuff_recieved_1_reshaped, 
        cuff_recieved_2_reshaped, cuff_recieved_3_reshaped, time_ticks, NUM_PULSES]
        hammer_times: Hammer times in ms. First col of hammer csv
        hammer_recieved: Hammer Arduino recieved voltages (V). Second col of hammer csv
        emg_recieved: EMG data in voltages (V), empty array if use_emg = False
        cuff_times_reshaped:[[times of recieved pulse 1], [times of recieved pulse 2],...]
        cuff_recieved_1_reshaped: [[voltages of recieved pulse 1], [voltages of recieved pulse 2],...]
        cuff_received_2_reshaped, cuff_received_3_reshaped: same as cuff_recieved_1 reshaped except
            for transducers 2 and 3 instead of transducer 1
        time_ticks: Used on heat maps -- the starting time of each pulse.
        NUM_PULSES: Number of recieved pulses detected in the cuff data. 
    '''
    my_csv = pd.read_csv(output_file, header=None).to_numpy()

    cuff_times = my_csv[:,0]
    cuff_recieved_1 = my_csv[:,1]
    cuff_recieved_2 = my_csv[:,2]
    cuff_recieved_3 = my_csv[:,3]
    hammer_recieved = my_csv[:,4]
    hammer_times = cuff_times
    emg_recieved = my_csv[:,5]
    
    NUM_PULSES = int(len(cuff_recieved_1) / DATA_LENGTH)
    print(f"{NUM_PULSES} recieved pulses found.")

    # Reshape by pulse (NUM_PULSES rows, DATA_LENGTH columns)
    cuff_times_reshaped = []
    cuff_recieved_1_reshaped = []
    cuff_recieved_2_reshaped = []
    cuff_recieved_3_reshaped = []
    time_ticks = []
    i = 0
    for r in range(NUM_PULSES):
        cuff_pulse_times = []
        cuff_pulse_data_1 = []
        cuff_pulse_data_2 = []
        cuff_pulse_data_3 = []
        
        for c in range(DATA_LENGTH):
            cuff_pulse_times.append(cuff_times[i])
            cuff_pulse_data_1.append(cuff_recieved_1[i])
            cuff_pulse_data_2.append(cuff_recieved_2[i])
            cuff_pulse_data_3.append(cuff_recieved_3[i])
            i+=1
        
        cuff_times_reshaped.append(cuff_pulse_times)
        cuff_recieved_1_reshaped.append(cuff_pulse_data_1)
        cuff_recieved_2_reshaped.append(cuff_pulse_data_2)
        cuff_recieved_3_reshaped.append(cuff_pulse_data_3)
        time_ticks.append(round(cuff_pulse_times[0], 2))

    return [hammer_times, hammer_recieved, emg_recieved, cuff_times_reshaped, cuff_recieved_1_reshaped, 
            cuff_recieved_2_reshaped, cuff_recieved_3_reshaped, time_ticks, NUM_PULSES]

def plot_heat_map(input_files, folder_path = files_folder_path, png_name = "cuff_hammer_emg_combined", stddev = 2, use_emg = False, normalize_to_initial = True):
    '''
    Plots hammer hit versus cuff heatmap, and allows user to select an area to search for the maximum intensity in. \n

    Inputs: \n
        input_files: [hammer_times, hammer_recieved, emg_recieved, cuff_times_reshaped, 
        cuff_recieved_1_reshaped, cuff_recieved_2_reshaped, cuff_recieved_3_reshaped, time_ticks, NUM_PULSES]
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
    start_index = 0
    end_index = int(len(input_files[0]))
    hammer_times = input_files[0][start_index:end_index]
    hammer_recieved = input_files[1][start_index:end_index]
    emg_recieved = input_files[2][start_index:end_index]
    cuff_times_reshaped = input_files[3][start_index:end_index]
    cuff_recieved_1_reshaped = input_files[4][start_index:end_index]
    cuff_recieved_2_reshaped = input_files[5][start_index:end_index]
    cuff_recieved_3_reshaped = input_files[6][start_index:end_index]
    time_ticks = input_files[7][start_index:end_index]
    NUM_PULSES = input_files[8]

    # Filter (smoothen heatplot)
    cuff_recieved_1_reshaped = uniform_filter1d(cuff_recieved_1_reshaped, size=20, axis=0, mode='nearest')
    cuff_recieved_2_reshaped = uniform_filter1d(cuff_recieved_2_reshaped, size=20, axis=0, mode='nearest')
    cuff_recieved_3_reshaped = uniform_filter1d(cuff_recieved_3_reshaped, size=20, axis=0, mode='nearest')
    cuff_recieved_reshaped = [cuff_recieved_1_reshaped, cuff_recieved_2_reshaped, cuff_recieved_3_reshaped]

    # Plot using GridSpec
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(5, 1, height_ratios=[0.5, 1, 1, 1, 0.05])

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

    # Color map: Set color/voltage limits for all 3 signals
    # Stack all data arrays to find global color scale
    all_cuff_data = np.array(cuff_recieved_reshaped)
    if normalize_to_initial:
        all_cuff_data = all_cuff_data - all_cuff_data[-1, :]
    # Concatenate all for joint outlier detection
    combined_data = np.concatenate(all_cuff_data, axis=0)
    lower_outliers, upper_outliers, global_vmin, global_vmax = find_outliers_std(combined_data, stddev)

    # Each transducer signal subplot
    im = None
    for i in range(len(cuff_recieved_reshaped)):
        ax2 = plt.subplot(gs[i+1])
        cuff_vals_for_heatmap = np.asarray(cuff_recieved_reshaped[i])
        if (normalize_to_initial): 
            cuff_vals_for_heatmap = np.asarray(cuff_vals_for_heatmap) - np.asarray(cuff_vals_for_heatmap)[-1, :]
        lower_outliers, upper_outliers, lower_lim_imshow, upper_lim_imshow = find_outliers_std(cuff_vals_for_heatmap, stddev)
        im = ax2.imshow(np.transpose(cuff_vals_for_heatmap), aspect='auto', cmap='jet', vmin=global_vmin, vmax=global_vmax)
        ax2.set_title(f'Transducer {i+1} Circuit envelope')
        ax2.set_ylabel('Arr. indx w/in pulse')
        ax2.set_xlabel('Start time of pulse (ms)')
        time_tick_positions = np.arange(0, NUM_PULSES, NUM_PULSES / len(time_ticks))
        ax2.set_xticks(ticks=time_tick_positions[0::150])
        ax2.set_xticklabels(labels=time_ticks[0::150])
        ax2.tick_params(axis='x', rotation=0)

    # Colorbar subplot
    cbar_ax = plt.subplot(gs[-1])
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

    parser.add_argument('--filename_suffix', type=str, help = 'appended to name of csvs (do not include .csv postfix)') 
    parser.add_argument('--folder_path', type=str, help='to change the default path (can insert full or relative path)', nargs='?') 
    parser.add_argument('--live_mode', action='store_true', help='Set read_live_data to True')
    parser.add_argument('--readback_mode', action='store_true', help='Set read_live_data to False') 

    args = parser.parse_args()
    if args.folder_path is not None: files_folder_path = args.folder_path
    if args.filename_suffix is not None: file_name = str(args.filename_suffix)
    if args.live_mode: read_live_data  = True
    if args.readback_mode: read_live_data  = False
    

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
    data_arrays = get_reshaped_array_from_arduino_csv(output_file, DATA_LENGTH)
    print(f"Reading {output_file}")
    plot_heat_map(data_arrays, png_name=file_name, folder_path=files_folder_path)
    #s'''
