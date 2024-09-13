'''
Name: analyze_Pico_data.py
Last updated: 9/12/24 by Ruth Berkun

Table of contents:
    Helper functions:
        find_maxima_envelope(signal, sampling_rate, approx_frequency):
            Returns the envelope optained by connecting the relative maxima of the signal
        butter_lowpass_filter(data, cutOff, fs, order=4):
            Lowpass filter, returns lowpass filtered data
        get_filtered_pulses(data, Sampling_frequency, use_raw_envelope = False):
            Returns filtered version of data
        get_envelope(data, use_raw_envelope = False):
            Returns envelope of data
        find_outliers_std(data, threshold=3):
            Used to set color limits on heat maps and axis limits on gif
        sum_signals(times1, voltages1, times2, voltages2):
            Sums two time-varying 1D signals
        check_time_unit(csv_file_name):   
            Checks whether our CSV is in seconds or milliseconds

    Functions to analyze cuff data:
        get_reshaped_arrays(all_csv_data, col_indexes):
            Outputs 2D array for cuff heatmap, as well as other processing information for other functions to use.
        get_phase_arrays(all_csv_data, col_indexes):
            Used to analyze phase shift of the signal over slow time. Not used currently
            and still in development.

    Functions to display and save data:
        plot_heat_map(input_files, folder_path = "", png_name = "", stddev = 3, use_emg = False, plot_circuit_env = False):
            Plots heat map of given 2D array, displays it, allows user to search for a maximum, and saves it to specified location
        plot_heat_map_fourier(input_files, folder_path = "", png_name = "", stddev = 1, in_ms = True):
            Plots heat map of the FFT of the signal (y axis frequency, x axis time) and saves it to the specified location
        get_gif(all_csv_data, col_indexes, save_as_mp4 = True, plot_hammer = False, plot_circuit_envelope = True, plot_calculated_envelope = True, compare_contraction = False, active_recieved_pulses_filtered = None, file_folder_name = "", specific_file_name = ""):
            Plot the recieved pulses over time as a GIF or mp4
        get_fourier_gif(input_files, col_indexes, save_as_mp4 = True, in_ms = True, file_folder_name = "", specific_file_name = ""):
            Plot the FFT of each pulse over time as a GIF or mp4
        plot_2d(input_file_array, legends, use_integral = True, use_calculated_env = True, use_abs = True, folder="", title="fig"):
            Plots a heatmap as a single line (the line being the intergal or average of each pulse.)

    Functions to do all the analysis on (a) trial(s) at once:
        analyze_one_trial(file_name, col_order, plot_circuit_envelope):
            Plots the heat map, 2D line, and frequency information for one trial
        summary_of_trials(file_names, col_order, experiment_name, analyze_circuit_env):
            Plots the average heat map and line plot for the experiment, as well as overlayed line plots for each trial 

Instructions for use: 
    - Stuff for user to edit is under 'if __name__ == "__main__":'
        Wherever you see a "# !!!!!!!!!!!!!!!", this is something you should pay attention to edit.
    - Once user is done editing the file location and function call parameters:
    Lastly, in the terminal, run python analyze_Pico_data.py
'''

####################################################################################################### Imports

import numpy as np
import matplotlib.pyplot as plt
import time as time
from scipy.signal import butter, sosfilt
from scipy.signal import hilbert
import pandas as pd
from scipy.signal import argrelextrema
from scipy.signal import medfilt
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks
import matplotlib.gridspec as gridspec
from matplotlib.widgets import RectangleSelector
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import decimate

verbose = False

####################################################################################################### Helper fns

def find_maxima_envelope(signal, sampling_rate, approx_frequency):
    """
    Find the envelope of a signal by identifying local maxima and connecting them with lines.

    Parameters:
    signal (numpy.ndarray): The input signal.
    sampling_rate (float): The sampling rate of the signal in Hz.
    approx_frequency (float): The approximate frequency of the signal in Hz.

    Returns:
    numpy.ndarray: The envelope of the input signal.
    """
    # Calculate the minimum distance between peaks in samples
    min_distance = int(sampling_rate / (approx_frequency/2))

    # Find the indices of the local maxima with the specified minimum distance
    peaks, _ = find_peaks(signal, distance=min_distance)

    # Get the values of the signal at the local maxima
    maxima = signal[peaks]

    # Interpolate between maxima to create the envelope
    envelope = np.interp(np.arange(len(signal)), peaks, maxima)

    return envelope

def butter_lowpass_filter(data, cutOff, fs, order=4):
    def butter_lowpass(cutoff, fs, order=5):
        return butter(order, cutoff, fs=fs, btype='low', analog=False)
    
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_filtered_pulses(data, Sampling_frequency, use_raw_envelope = False, filter_frequencies = [40000, 60000]):
    '''
    Filters the inputted signal.

    Inputs:
    - data: array of voltages of the signal to filter
    - Sampling frequency: How many Hz data was sampled at
    - filter_frequencies: Frequencies the bandpass filter keeps.
    - use_raw_envelope: If true, do median filtering, otherwise do bandpass/lowpass filter.

    Outputs:
    - Array of filtered voltages.
    '''

    lower_border = 10           # if filter_frequencies[0] < lower_border, it's a lowpass 
    higher_border = 1000000     # if filter_frequencies[1] > higher_border, do not filter.
    data = np.asarray(data, dtype=float)

    if (use_raw_envelope):
        kernel_size = 3
        return medfilt(data, kernel_size)
    else:
        Filter_lowcut = filter_frequencies[0]
        Filter_highcut = filter_frequencies[1]
        Filter_order = 4        # not really sure what this is.

        if filter_frequencies[1] > higher_border: return data    # no filtering.

        if filter_frequencies[0] > lower_border:
            sos = butter(Filter_order, [Filter_lowcut, Filter_highcut], btype='bandpass', fs=Sampling_frequency, output='sos')
            filtered= np.apply_along_axis(lambda x: sosfilt(sos, x), axis=0, arr = data)
        else:
            window_size_ms = 0.5
            dt = 1000/Sampling_frequency   # in ms
            window_size = int(window_size_ms/dt)
            # Ensure the window size is odd (required for median filter)
            if window_size % 2 == 0:
                window_size += 1
            filtered = medfilt(data, window_size)
            # print(f"filtered data length {len(filtered)}, versus original {len(data)}, window size {window_size}")
            
        if (verbose):
            plt.plot(np.arange(len(data)), data)
            plt.plot(np.arange(len(filtered)), filtered)
            plt.show()
        return filtered

def get_envelope(data, use_raw_envelope = False):
    '''
    Get the envelope of the signal.

    Inputs:
    - data: array of voltages of the signal to filter
    - use_raw_envelope: If true, use maxima envelope, otherwise use smooth Hilbert envelope.

    Outputs:
    - array of the envelope voltages
    '''
    if use_raw_envelope:
        lowest_freq_of_recieved_sig = 20000 # needs to be much lower than 52 kHz -- this is what determines the distance between points on the envelope
        osc_sampling_freq = 200000
        calculated_envelope = find_maxima_envelope(data, osc_sampling_freq, lowest_freq_of_recieved_sig)
    else:
        calculated_envelope = np.abs(hilbert(data, axis=0))

    return calculated_envelope

def find_outliers_std(data, threshold=3):
    '''
    Used to set color limits on heat maps and axis limits on gif.\n

    Inputs: 
    - data: Data to find outliers of
    - threshold: How many standard deviations away from the mean is considered an outlier.

    Outputs:
    - lower_outliers: list of lower outlier voltages
    - upper_outliers: list of upper outlier voltages
    - lower_bound: Smallest number that is not an outlier
    - upper_bound: Largerst number that is not an outlier
    '''
    # Ensure data is a NumPy array
    data = np.asarray(data)

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

def sum_signals(times1, voltages1, times2, voltages2):
    # Step 1: Find the common time range
    start_time = max(times1[0], times2[0])
    end_time = min(times1[-1], times2[-1])
    
    # Create a common time array with enough resolution (adjust as needed)
    common_times = np.linspace(start_time, end_time, num=500)  # Change num for resolution

    # Step 2: Interpolate voltages for both signals at common time points
    interp_voltages1 = interp1d(times1, voltages1, kind='linear', bounds_error=False, fill_value=0)
    interp_voltages2 = interp1d(times2, voltages2, kind='linear', bounds_error=False, fill_value=0)
    
    # Resample both signals at the common time points
    voltages1_resampled = interp_voltages1(common_times)
    voltages2_resampled = interp_voltages2(common_times)
    
    # Step 3: Sum the voltages
    summed_voltages = voltages1_resampled + voltages2_resampled
    
    return common_times, summed_voltages

def check_time_unit(csv_file_name):
    '''
    Check if the time units of our CSV is in ms or seconds. 
    Returns time unit as string "(ms)" or "(s)"
    '''

    df = pd.read_csv(csv_file_name, nrows=2, sep=',' if csv_file_name[-1]=="v" else "\s+")  # Read only the first two rows
    time_unit = df.iloc[0,0]  # Get the unit from the second row
    return time_unit

####################################################################################################### Processing functions

def get_reshaped_arrays(all_csv_data, col_indexes, filter_freqs = [40000, 60000], in_ms = True, keep_raw_data = True):
    '''
    Outputs 2D array for cuff heatmap, as well as other processing information for other functions to use. \n

    Input: \n
    - all_csv_data: your csv data as a pandas dataframe
    - col_indexes: Tells us which column in your CSV corresponds to what data collected from oscilloscope. \n
    \t col_indexes = [times_col_index, hammer_index, transmit_col_index, recieve_col_index, circuit_col_index, emg_col_index = 1]
    (It tells us which column of the CSV corresponds to times, to hammer, to cuff, etc)
    - keep_raw_data: If False, in the output, cuff_recieved_reshaped is the filtered pulses reshaped. If True, it is the raw pulses reshaped
                    Usually only set to False for when we're lowpassing (when we want cuff_recieved_reshaped to be the lowpassed signal)
    
    Return: hammer_times, hammer_recieved, emg_recieved, cuff_times_reshaped, cuff_recieved_reshaped, circuit_env_reshaped, time_ticks, NUM_PULSES
    In each reshaped array: one row is one pulse. Reshaped arrays are 2D and all other arrays are 1D.
    - hammer_times, hammer_recieved, emg: Times and voltages of the hammer and EMG signal from the CSV.
    - cuff_times_reshaped: The CSV times reshaped in the same way the recieved signal is reshaped.
    - cuff_recieved_reshaped, circuit_env_reshaped: Filtered recieved pulses and circuit envelope pulses reshaped, respectively.
    - time_ticks: Start time of each pulse (1D array)
    '''
    use_raw_envelope = False if filter_freqs[1] < 1000000 else True   # If False, Hilbert envelope, if True, maxima envelope 
                                                                    # Maxima envelope should be used for unfiltered signal    
    
    # This is dependent on the system, and used to organize where each recieved pulse begins and ends
    SQUARE_PULSES_EXPECTED = 10
    LENGTH_ONE_SQUARE_PULSE_MS = 1000.0/52000
    
    # Extract CSV information.
    csv_times = all_csv_data[:,col_indexes[0]]
    csv_square_pulses = all_csv_data[:,col_indexes[2]]
    csv_recieved_pulses = all_csv_data[:,col_indexes[3]]
    csv_circuit_envelope =  all_csv_data[:,col_indexes[4]]
    csv_hammer = all_csv_data[:,col_indexes[1]]
    emg_recieved = all_csv_data[:, col_indexes[5]]

    dt = np.mean(np.diff(csv_times))
    if (not in_ms): dt = dt*1000   # Convert seconds to ms
    fs = 1000 / dt # Sampling frequency, Hz
    ############################################

    hammer_times, hammer_recieved = csv_times, csv_hammer

    filtered_recieved_pulses = get_filtered_pulses(csv_recieved_pulses, fs, use_raw_envelope, filter_frequencies=filter_freqs)
    calculated_envelope = get_envelope(filtered_recieved_pulses, use_raw_envelope)
    if (keep_raw_data): filtered_recieved_pulses = csv_recieved_pulses

    NUM_PULSES = 0

    #####################################################  get reshaping scheme from transmit pulse
    total_start_indicies =  []
    i = 0
    r = 0
    tr = -5
    max_pulse_length_in_indicies = 0
    while i < (len(csv_square_pulses) - 1):
        if (csv_square_pulses[i] < tr and csv_square_pulses[i+1] >= tr):
            # print(f"{r}, {i}, {csv_square_pulses[i]}, {csv_square_pulses[i+1]}")
            total_start_indicies.append(i)
            if (r > 0):
                max_pulse_length_in_indicies = max(max_pulse_length_in_indicies, total_start_indicies[r] - total_start_indicies[r-1])
            r += 1
            i += int((LENGTH_ONE_SQUARE_PULSE_MS * (SQUARE_PULSES_EXPECTED + 1))/dt)     # skip to next pulse burst
        i+=1
    # print(max_pulse_length_in_indicies)
    NUM_PULSES = r

    # Check if we should skip the first pulse
    first_pulse = csv_square_pulses[total_start_indicies[0]:total_start_indicies[1]]
    peaks_found = 0
    i = 0
    while i < len(first_pulse) - 1:
        if (first_pulse[i] < tr and first_pulse[i+1] >= tr):
            peaks_found += 1
            i += int((LENGTH_ONE_SQUARE_PULSE_MS * 0.6)/dt)     # skip to next square
        i+=1
    if peaks_found < 10: 
        total_start_indicies.remove(total_start_indicies[0])
        NUM_PULSES -= 1


    #################################### Reshaped arrays . shorter pulses padded with the last data point.
    times_reshaped = np.zeros((NUM_PULSES, max_pulse_length_in_indicies))
    square_pulses_reshaped = np.zeros((NUM_PULSES, max_pulse_length_in_indicies))
    calculated_envelope_reshaped = np.zeros((NUM_PULSES, max_pulse_length_in_indicies))
    circuit_envelope_reshaped = np.zeros((NUM_PULSES, max_pulse_length_in_indicies))
    recieved_pulses_reshaped = np.zeros((NUM_PULSES, max_pulse_length_in_indicies))

    r = 0
    for s in range(len(total_start_indicies)):
        this_start = total_start_indicies[s]
        next_start = len(csv_times) - 1
        if (s + 1 < len(total_start_indicies)): next_start = total_start_indicies[s+1]

        for c in range(max_pulse_length_in_indicies):

            if (c + this_start < next_start):
                times_reshaped[r][c] = csv_times[c + this_start]
                square_pulses_reshaped[r][c] = csv_square_pulses[c + this_start]
                calculated_envelope_reshaped[r][c] = calculated_envelope[c + this_start]
                circuit_envelope_reshaped[r][c] = csv_circuit_envelope[c + this_start]
                recieved_pulses_reshaped[r][c] = filtered_recieved_pulses[c + this_start]
            else:
                times_reshaped[r][c] = csv_times[next_start - 1]
                square_pulses_reshaped[r][c] = csv_square_pulses[next_start - 1]
                calculated_envelope_reshaped[r][c] = calculated_envelope[next_start - 1]
                circuit_envelope_reshaped[r][c] = csv_circuit_envelope[next_start - 1]
                recieved_pulses_reshaped[r][c] = filtered_recieved_pulses[next_start - 1]

        r += 1

    start = np.zeros(NUM_PULSES)

    ''' 
    # 10 pulses lasts about 200 indexes
    # Sanity check that we are slicing the pulses correctly
    for r in range(NUM_PULSES):
        fig = plt.figure(figsize =(10, 1))
        plt.plot(times_reshaped[r], square_pulses_reshaped[r])
        plt.show()
    # '''

    points_per_col = max_pulse_length_in_indicies - 1
    ########################################################################## Get time ticks
    time_ticks = []
        
    for i in range(NUM_PULSES - 1):
        time_ticks.append(round(times_reshaped[i, start[i].astype(int)], 2)) # Start time of each pulse

    #print("Time ticks: ")
    #print(time_ticks)

    return hammer_times, hammer_recieved, emg_recieved, times_reshaped, recieved_pulses_reshaped, circuit_envelope_reshaped, calculated_envelope_reshaped, time_ticks, NUM_PULSES

def get_phase_arrays(all_csv_data, col_indexes):
    '''
    Used to analyze phase shift of the signal over slow time. Not used currently
    and still in development.
    '''
    #################################################################################### Quadrature phase stuff
    ############################################# Quadrature modulation time! ###########################################
    #####################################################################################################################
    #####################################################################################################################

    hammer_times, hammer_recieved, emg_recieved, times_reshaped, recieved_pulses_reshaped, circuit_envelope_reshaped, calculated_envelope_reshaped, time_ticks, NUM_PULSES = get_reshaped_arrays(all_csv_data, col_indexes)
    csv_times = all_csv_data[:,col_indexes[0]]
    csv_square_pulses = all_csv_data[:,col_indexes[2]]
    csv_recieved_pulses = all_csv_data[:,col_indexes[3]]
    csv_circuit_envelope =  all_csv_data[:,col_indexes[4]]
    csv_hammer = all_csv_data[:,col_indexes[1]]
    emg_recieved = all_csv_data[:, col_indexes[5]]

    Filter_lowcut = 1
    Filter_highcut = 40000
    Filter_order = 4
    Sampling_frequency=1000/(csv_times[1]-csv_times[0]) # in Hz, since time is in ms
    lowpass_sin_cos_ref = butter(Filter_order, [Filter_lowcut, Filter_highcut], btype='bandpass', fs=Sampling_frequency, output='sos')


    #################################################################################### Step 1: Find dominant frequency

    show_recieved_fft_plot = False

    dt = np.mean(np.diff(csv_times))
    fs = 1000 / dt  # Sampling frequency, in Hz

    # Compute FFT
    fft_values = np.fft.fft(csv_recieved_pulses)
    fft_frequencies = np.fft.fftfreq(len(csv_recieved_pulses), dt)

    # Only take the positive frequencies (and corresponding FFT values)
    positive_frequencies = fft_frequencies[:len(fft_frequencies)//2]
    positive_fft_values = fft_values[:len(fft_values)//2]

    peak_frequency_index = np.argmax(np.abs(positive_fft_values))
    peak_frequency = positive_frequencies[peak_frequency_index]

    # Plotting
    if show_recieved_fft_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(positive_frequencies, np.abs(positive_fft_values))
        plt.title('FFT of the Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid()
        plt.show()

    # convolute with boxcar, length of the 10 transmitted pulses in slow time
    # someday, if need to speed up timing: inv_fft(FFT(box) * FFT(sin_ref))

    def ruth_convolution(waveform_1):
        conv_waveform = np.zeros(len(waveform_1))
        len_boxcar = 200
        for i in range(len(waveform_1)):
            for j in range(len_boxcar):
                if (i - j < len(waveform_1) and i - j > 0): 
                    conv_waveform[i] += waveform_1[i-j] 
        return conv_waveform

    recieved_freq = peak_frequency
    #################################################################################### Step 2: Mixing 

    phase_shift = []
    phase_shift_times = []
    phase_reshaped = []
    for r in range(NUM_PULSES):

        ############################################################### Convert to baseband
        times_to_conv = times_reshaped[r]
        sin_ref_fast_time = np.sin(2*np.pi*recieved_freq*(times_to_conv - times_to_conv[0])) * recieved_pulses_reshaped[r]
        cos_ref_fast_time = np.cos(2*np.pi*recieved_freq*(times_to_conv - times_to_conv[0])) * recieved_pulses_reshaped[r]

        ################################################################ Low pass filter the sine wave
        sin_ref_fast_time =  np.apply_along_axis(lambda x: sosfilt(lowpass_sin_cos_ref, x), axis=0, arr = sin_ref_fast_time)
        cos_ref_fast_time = np.apply_along_axis(lambda x: sosfilt(lowpass_sin_cos_ref, x), axis=0, arr = cos_ref_fast_time)

        ################################################################ Convolve with boxcar (Pulse compression)
        sin_ref_fast_time = ruth_convolution(sin_ref_fast_time)
        cos_ref_fast_time = ruth_convolution(cos_ref_fast_time)

        ################################################################ Arctangent to get phase shift
        phase_shift_fast_time = np.arctan2(sin_ref_fast_time, cos_ref_fast_time)
        # plt.plot(times_to_conv, phase_shift_fast_time)
        # plt.show()

        ################################################################ Add to 1D array and 2D array
        phase_reshaped.append(phase_shift_fast_time)
        for c in range(len(phase_shift_fast_time)):
            phase_shift.append(phase_shift_fast_time[c])
            phase_shift_times.append(times_to_conv[c])

    plt.title("Phase shift versus slow time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Phase shift (radians)")
    plt.plot(phase_shift_times, phase_shift)
    phase_reshaped = np.array(phase_reshaped)

    for c in range(len(phase_reshaped[0])):
        ################################################################ Filter along range line
        my_col = phase_reshaped[:,c]
        phase_reshaped[:,c] = medfilt(my_col, 9)

    return phase_reshaped

####################################################################################################### Plotting functions

# Plots

def plot_heat_map(input_files, folder_path = "", png_name = "", stddev = 3, use_emg = False, plot_circuit_env = False, in_ms = True):
    '''
    Plots heat map of given 2D array, displays it, allows user to search for a maximum, and saves it to specified location.\n

    Inputs:
    - input_files: Same format as output of get_reshaped_arrays.
        hammer_times, hammer_recieved, emg_recieved, cuff_times_reshaped, cuff_recieved_reshaped, circuit_env_reshaped, calculated_envelope_reshaped, time_ticks, NUM_PULSES
    - folder_path: Folder to save heatmap in. Must NOT end in "/". If it and png_name are blank, image will not be saved.
    - png_name: Name of image to save heatmap in. If it and folder_path are blank, image will not be saved.
    - stddev: How many standard deviations from the mean will be considered an outlier. Outliers determine the color limits
            of the heat map. Make it lower to see more intense colors overall, and higher to see less intense colors overall.
    - use_emg: Boolean telling us whether or not to plot the EMG signal on top of the hammer signal.
    - plot_circuit_env: If true, the heat map will be of the circuit envelope. If false, the heat map will be of the calculated envelope.
    - in_ms: True if the time unit of the CSV is in ms, False if it is in seconds

    Ouputs:
    - Displays (and, if specified, saves) heatmap of the passed input arrays.
    '''

    # Retrieve the data we need for the heat map
    hammer_times = input_files[0]
    hammer_recieved = input_files[1]
    emg_recieved = input_files[2]
    cuff_times_reshaped = input_files[3]
    cuff_recieved_reshaped = input_files[4]
    circuit_env_reshaped = input_files[5]
    calculated_envelope_reshaped = input_files[6]
    time_ticks = input_files[7]
    NUM_PULSES = input_files[8]

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
    title_addend = "Circuit" if plot_circuit_env else "Calculated"
    ax2 = plt.subplot(gs[1])
    cuff_vals_for_heatmap = np.asarray(calculated_envelope_reshaped) - np.asarray(calculated_envelope_reshaped)[0, :]
    if plot_circuit_env: 
        cuff_vals_for_heatmap = np.asarray(circuit_env_reshaped) - np.asarray(circuit_env_reshaped)[0, :]
    lower_outliers, upper_outliers, lower_lim_imshow, upper_lim_imshow = find_outliers_std(cuff_vals_for_heatmap, stddev)
    im = ax2.imshow(np.transpose(cuff_vals_for_heatmap), aspect='auto', cmap='jet', vmin=lower_lim_imshow, vmax=upper_lim_imshow)
    ax2.set_title(title_addend+' envelope: \nPulse height vs time normalize to start of pulse, all pulses overlayed')
    ax2.set_ylabel('Array index within pulse')
    ax2.set_xlabel('Start time of pulse (ms)') if in_ms else ax2.set_xlabel('Start time of pulse (s)')
    
    time_tick_positions = np.arange(1, NUM_PULSES, NUM_PULSES / len(time_ticks))
    selected_positions = time_tick_positions[0::int(len(time_ticks)/10)]
    selected_ticks = np.round(np.asarray(time_ticks[0::int(len(time_ticks)/10)]), 2)
    selected_positions = selected_positions[:min(len(selected_ticks), len(selected_positions))]
    selected_ticks = selected_ticks[:min(len(selected_ticks), len(selected_positions))]
    ax2.set_xticks(ticks=selected_positions)
    ax2.set_xticklabels(labels=selected_ticks)
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

    str_name = folder_path + "/" + png_name + '_'+title_addend+'_envelope.png'
    if len(str_name) > 5: fig.savefig(str_name)
    print(f"Saving to: {str_name}")
    plt.close(fig)
    
    # Return time of reflex
    min_reflex_time = 28 # in ms
    max_reflex_time = 50 # in ms
    min_reflex_index_within_pulse = 30
    reflex_time = 0
    max_amplitude_heat_map = 0
    # print(f"NUM_PULSES: {NUM_PULSES}")
    for r in range(NUM_PULSES):
        for c in range(min_reflex_index_within_pulse, len(cuff_recieved_reshaped[r])):
            amplitude_diff = cuff_recieved_reshaped[r][c] - cuff_recieved_reshaped[0][c]
            if (cuff_times_reshaped[r][c] < min_reflex_time or cuff_times_reshaped[r][c] > max_reflex_time): continue
            if np.abs(amplitude_diff) > max_amplitude_heat_map and amplitude_diff > lower_lim_imshow and amplitude_diff < upper_lim_imshow:
                max_amplitude_heat_map = np.abs(amplitude_diff)
                reflex_time = cuff_times_reshaped[r][c]
    # print(f"Auto-detect found: Maximum muscle contraction found at {reflex_time} ms after hammer hit.")

def plot_heat_map_fourier(input_files, folder_path = "", png_name = "", stddev = 1, in_ms = True):
    '''
    Plots Fourier heat map of given 2D array, displays it, allows user to search for a maximum, and saves it to specified location.\n

    Inputs:
    - input_files: Same format as output of get_reshaped_arrays.
        hammer_times, hammer_recieved, emg_recieved, cuff_times_reshaped, cuff_recieved_reshaped, circuit_env_reshaped, calculated_envelope_reshaped, time_ticks, NUM_PULSES
    - folder_path: Folder to save heatmap in. Must NOT end in "/". If it and png_name are blank, image will not be saved.
    - png_name: Name of image to save heatmap in. If it and folder_path are blank, image will not be saved.
    - stddev: How many standard deviations from the mean will be considered an outlier. Outliers determine the color limits
            of the heat map. Make it lower to see more intense colors overall, and higher to see less intense colors overall.
    - in_ms: True if the time unit of the CSV is in ms, False if it is in seconds
    
    Ouputs:
    - Displays (and, if specified, saves) heatmap of the passed input arrays.
    '''
    show_zoomed_fourier_maps = False

    # Retrieve the data we need for the heat map
    hammer_times = input_files[0]
    hammer_recieved = input_files[1]
    emg_recieved = input_files[2]
    cuff_times_reshaped = input_files[3]
    cuff_recieved_reshaped = input_files[4]
    circuit_env_reshaped = input_files[5]
    calculated_envelope_reshaped = input_files[6]
    time_ticks = input_files[7]
    NUM_PULSES = input_files[8]

    # Get the FFT plots.
    dt = np.mean(np.diff(hammer_times))
    if (not in_ms): dt = dt*1000
    # print(f"sampling frequency {1000/dt} Hz")

    # Compute FFT
    fft_arr = []
    positive_fft_frequencies = []
    for recieved_pulse in cuff_recieved_reshaped:

        fft_values = np.fft.fft(recieved_pulse)
        fft_frequencies = np.fft.fftfreq(len(recieved_pulse), dt)

        # Only take the positive frequencies (and corresponding FFT values)
        positive_fft_frequencies = fft_frequencies[:len(fft_frequencies)//2]    # the x axis, frequency
        positive_fft_values = np.abs(fft_values[:len(fft_values)//2])       # the y axis, number per frequency bucket
        fft_arr.append(positive_fft_values)

    # print(f"{positive_fft_frequencies[0]} kHz to {positive_fft_frequencies[-1]} kHz, length {len(positive_fft_frequencies)}")

    # Plot using GridSpec
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.05])

    # Hammer and EMG signal subplot
    ax1 = plt.subplot(gs[0])
    ax1.plot(hammer_times, hammer_recieved, color="blue", label="Hammer strike")
    ax1.set_xlim(time_ticks[0], time_ticks[-1])
    ax1.set_title("Hammer voltage vs time")
    ax1.set_ylabel('Voltage (V)')
    ax1.set_xlabel('Time (ms)')
    ax1.legend()

    # Cuff signal subplot
    ax2 = plt.subplot(gs[1])
    cuff_vals_for_heatmap = fft_arr
    lower_outliers, upper_outliers, lower_lim_imshow, upper_lim_imshow = find_outliers_std(cuff_vals_for_heatmap, stddev)
    lower_lim_imshow = 0
    im = ax2.imshow(np.transpose(cuff_vals_for_heatmap), aspect='auto', cmap='jet', vmin=lower_lim_imshow, vmax=upper_lim_imshow)
    ax2.set_title("Calculated FFT")
    
    # Setting y-axis ticks and labels
    ax2.set_ylabel('Frequency (kHz)')
    selected_frequency_ticks = np.round(np.asarray(positive_fft_frequencies[0::10]),2)
    ax2.set_yticks(selected_frequency_ticks)
    ax2.set_ylim(0, min(80, positive_fft_frequencies[-1])) # limit to 80 kHz

    # Setting x-axis ticks and labels
    time_tick_positions = np.arange(1, NUM_PULSES, NUM_PULSES / len(time_ticks))
    selected_positions = time_tick_positions[0::int(len(time_ticks)/10)]
    selected_ticks = np.round(np.asarray(time_ticks[0::int(len(time_ticks)/10)]), 2)
    selected_positions = selected_positions[:min(len(selected_ticks), len(selected_positions))]
    selected_ticks = selected_ticks[:min(len(selected_ticks), len(selected_positions))]
    ax2.set_xticks(ticks=selected_positions)
    ax2.set_xticklabels(labels=selected_ticks)
    ax2.tick_params(axis='x', rotation=90)
    ax2.set_xlabel('Start time of pulse (ms)') if in_ms else ax2.set_xlabel('Start time of pulse (s)')

    # Colorbar subplot
    cbar_ax = plt.subplot(gs[2])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

    # Save the figure before showing it
    plt.subplots_adjust(hspace=1)
    plt.show()

    str_name = folder_path + "/" + png_name + '_FULL_COLORMAP.png'
    if len(str_name) > 5: fig.savefig(str_name)
    print(f"Saving to: {str_name}")

    if show_zoomed_fourier_maps:
        ################################################################################################# PLOT JUST THE LOWER PART
        # Plot using GridSpec
        fig = plt.figure(figsize=(6, 8))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.05])

        # Hammer and EMG signal subplot
        ax1 = plt.subplot(gs[0])
        ax1.plot(hammer_times, hammer_recieved, color="blue", label="Hammer strike")
        ax1.set_xlim(time_ticks[0], time_ticks[-1])
        ax1.set_title("Hammer voltage vs time")
        ax1.set_ylabel('Voltage (V)')
        ax1.set_xlabel('Time (ms)')
        ax1.legend()

        # Cuff signal subplot
        ax2 = plt.subplot(gs[1])
        cuff_vals_for_heatmap = fft_arr
        lower_lim_imshow = 0
        upper_lim = upper_lim_imshow / 5
        im = ax2.imshow(np.transpose(cuff_vals_for_heatmap), aspect='auto', cmap='jet', vmin=lower_lim_imshow, vmax=upper_lim)
        ax2.set_title("Calculated FFT")
        
        # Setting y-axis ticks and labels
        ax2.set_ylabel('Frequency (kHz)')
        selected_frequency_ticks = np.round(np.asarray(positive_fft_frequencies[0::10]),2)
        ax2.set_yticks(selected_frequency_ticks)
        ax2.set_ylim(0, 10) # limit to 80 kHz

        # Colorbar subplot
        cbar_ax = plt.subplot(gs[2])
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

        # Save the figure before showing it
        plt.subplots_adjust(hspace=1)
        plt.show()

        str_name = folder_path + "/" + png_name + '_LOWER_COLORMAP.png'
        if len(str_name) > 5: fig.savefig(str_name)
        print(f"Saving to: {str_name}")

        ############################################################################### PLOT JUST THE UPPER PART
            # Plot using GridSpec
        fig = plt.figure(figsize=(6, 8))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.05])

        # Hammer and EMG signal subplot
        ax1 = plt.subplot(gs[0])
        ax1.plot(hammer_times, hammer_recieved, color="blue", label="Hammer strike")
        ax1.set_xlim(time_ticks[0], time_ticks[-1])
        ax1.set_title("Hammer voltage vs time")
        ax1.set_ylabel('Voltage (V)')
        ax1.set_xlabel('Time (ms)')
        ax1.legend()

        # Cuff signal subplot
        ax2 = plt.subplot(gs[1])
        cuff_vals_for_heatmap = fft_arr
        lower_lim_imshow = 0
        upper_lim = upper_lim_imshow / 2
        im = ax2.imshow(np.transpose(cuff_vals_for_heatmap), aspect='auto', cmap='jet', vmin=lower_lim_imshow, vmax=upper_lim)
        ax2.set_title("Calculated FFT")
        
        # Setting y-axis ticks and labels
        ax2.set_ylabel('Frequency (kHz)')
        selected_frequency_ticks = np.round(np.asarray(positive_fft_frequencies[0::10]),2)
        ax2.set_yticks(selected_frequency_ticks)
        ax2.set_ylim(40, min(70, positive_fft_frequencies[-1])) # limit to 80 kHz

        # Colorbar subplot
        cbar_ax = plt.subplot(gs[2])
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

        # Save the figure before showing it
        plt.subplots_adjust(hspace=1)
        plt.show()

        str_name = folder_path + "/" + png_name + '_UPPER_COLORMAP.png'
        if len(str_name) > 5: fig.savefig(str_name)
        print(f"Saving to: {str_name}")

    plt.close(fig)

def get_gif(all_csv_data, col_indexes, save_as_mp4 = True, plot_hammer = False, plot_circuit_envelope = True, plot_calculated_envelope = True, compare_contraction = False, active_recieved_pulses_filtered = None, file_folder_name = "", specific_file_name = ""):
    '''
    Plot the recieved pulses over time as a GIF or mp4.\n

    Inputs:
    - all_csv_data: your csv data as a pandas dataframe
    - col_indexes: Tells us which column in your CSV corresponds to what data collected from oscilloscope!
        col_indexes = [times_col_index, hammer_index, transmit_col_index, recieve_col_index, circuit_col_index, emg_col_index = 1]
            (It tells us which column of the CSV corresponds to times, to hammer, to cuff, etc)
    Optional arguments:
    - save_as_mp4: If true, saves as MP4, if false, saves as GIF. You must have ffmeg installed to save as an MP4.
    - plot_hammer: Set true to plot the hammer signal in the gif
    - plot_circuit_envelope: Set true to plot the circuit envelope in the gif
    - plot_circuit_envelope: Set true to plot the calculated envelope in the gif
    - compare_contraction: Set true to plot the contracted signal in the gif as well (in the background). Not available currently.
    - active_recieved_pulses_filtered: The active recieved pulses (contracted signal) to plot in the background.
    - file_folder_name: Folder to save gif in. Must NOT end in "/". 
    - specific_file_name: Name of gif to save. 
    '''
    save_as_mp4 = True # Saves as GIF when this is false. You need ffmeg installed to save as mp4.
    use_raw_envelope = False
    csv_times = all_csv_data[:,col_indexes[0]]
    csv_square_pulses = all_csv_data[:,col_indexes[2]]
    csv_recieved_pulses = all_csv_data[:,col_indexes[3]]
    csv_circuit_envelope =  all_csv_data[:,col_indexes[4]]
    csv_hammer = all_csv_data[:,col_indexes[1]]
    emg_recieved = all_csv_data[:, col_indexes[5]]

    hammer_times, hammer_recieved, emg_recieved, times_reshaped, recieved_pulses_reshaped, circuit_envelope_reshaped, calculated_envelope_reshaped, time_ticks, NUM_PULSES = get_reshaped_arrays(all_csv_data, col_indexes)

    #######################################################################################################

    lower_outliers, upper_outliers, min_y_axis, max_y_axis = find_outliers_std(np.asarray(recieved_pulses_reshaped), 10)
    scale_circuit_envelope = max_y_axis*1.0/max(csv_circuit_envelope)
    scale_hammer = max_y_axis*1.0/max(csv_hammer)

    shift_active_up =  np.mean(np.asarray(recieved_pulses_reshaped)) - np.mean(active_recieved_pulses_filtered) if compare_contraction else 0
    shift_calc_env_up = 1 + np.mean(np.asarray(recieved_pulses_reshaped)) - np.mean(np.asarray(calculated_envelope_reshaped))

    def update(frame):
        plt.cla()  # Clear the current axes
        plt.plot(times_reshaped[frame] - times_reshaped[frame][0], recieved_pulses_reshaped[frame], label=f'Filtered recieved signal', alpha=0.7)
        if (plot_circuit_envelope): 
            plt.plot(times_reshaped[frame] - times_reshaped[frame][0], circuit_envelope_reshaped[frame]*scale_circuit_envelope, label=f'Circuit envelope', alpha=0.7)
        if (plot_calculated_envelope):
            plt.plot(times_reshaped[frame] - times_reshaped[frame][0], calculated_envelope_reshaped[frame]+shift_calc_env_up, label=f'Calculated envelope. shifted {shift_calc_env_up:.2f} mV up', alpha=0.7)
        '''
        if (plot_hammer):
            plt.plot(times_reshaped[frame] - times_reshaped[frame][0], hammer_signal_reshaped[frame]*scale_hammer, label=f'Hammer (scaled {scale_hammer:.2f})', alpha=0.7)
        if (compare_contraction):
            plt.plot(active_times_reshaped[frame] - active_times_reshaped[frame][0], active_recieved_pulses_reshaped[frame]+shift_active_up, label=f'Active contraction pulse, shifted {shift_active_up:.2f} mV up', alpha=0.3)
        '''
        plt.legend(loc='upper right')
        plt.xlim(0, times_reshaped[0][-1] - times_reshaped[0][0])
        plt.ylim(min_y_axis, max_y_axis)
        plt.title(f'Pulse {frame}, starts at {times_reshaped[frame][0]} ms since hammer hit')
        plt.xlabel('Time (ms) within pulse')
        plt.ylabel('Amplitude (mV)')

    fig, ax = plt.subplots(figsize=(10, 5))
    ani = FuncAnimation(fig, update, frames=NUM_PULSES, repeat=False)

    # Save as GIF
    extension = ".mp4" if save_as_mp4 else ".gif"
    save_place = file_folder_name + "/" + specific_file_name + '_pulse_animation' + extension
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800) if save_as_mp4 else PillowWriter(fps=10)
    ani.save(save_place, writer)
    print(f"Saved at: {save_place}")
    plt.show()

def get_fourier_gif(input_files, col_indexes, save_as_mp4 = True, in_ms = True, file_folder_name = "", specific_file_name = ""):
    '''
    Plot the FFT of each recieved pulse over time as a GIF or mp4.\n

    Inputs:
    - input_files: output of get_reshaped_arrays
    - col_indexes: Tells us which column in your CSV corresponds to what data collected from oscilloscope!
        col_indexes = [times_col_index, hammer_index, transmit_col_index, recieve_col_index, circuit_col_index, emg_col_index = 1]
            (It tells us which column of the CSV corresponds to times, to hammer, to cuff, etc)
    Optional arguments:
    - save_as_mp4: If true, saves as MP4, if false, saves as GIF. You must have ffmeg installed to save as an MP4.
    - file_folder_name: Folder to save gif in. Must NOT end in "/". 
    - specific_file_name: Name of gif to save. 
    - in_ms: True if the time unit of the CSV is in ms, False if it is in seconds

    Outputs: A GIF or MP4 of the FFT of the signal over time
    '''
    
    # Retrieve the data we need for the heat map
    hammer_times = input_files[0]
    hammer_recieved = input_files[1]
    emg_recieved = input_files[2]
    times_reshaped = input_files[3]
    recieved_pulses_reshaped = input_files[4]
    circuit_env_reshaped = input_files[5]
    calculated_envelope_reshaped = input_files[6]
    time_ticks = input_files[7]
    NUM_PULSES = input_files[8]

    # Get the FFT plots.
    dt = np.mean(np.diff(hammer_times)) * 0.001
    if (not in_ms): dt = dt*1000
    fs = 1/dt

    # Compute FFT
    fft_arr = []
    positive_fft_frequencies = []
    for recieved_pulse in recieved_pulses_reshaped:
        
        # Zero pad
        padding_length = 5000
        new_recieved_pulse = np.pad(recieved_pulse, (0, padding_length), 'constant')

        fft_values = np.fft.fft(new_recieved_pulse)
        fft_frequencies = np.fft.fftfreq(len(new_recieved_pulse), dt)
        # print(f"{fft_frequencies[0]} kHz to {fft_frequencies[-1]}, length {len(fft_frequencies)//2}")

        # Only take the positive frequencies (and corresponding FFT values)
        positive_fft_frequencies = fft_frequencies[:len(fft_frequencies)//2]    # the x axis, frequency
        positive_fft_values = np.abs(fft_values[:len(fft_values)//2])       # the y axis, number per frequency bucket
        fft_arr.append(positive_fft_values)

    plt.plot(positive_fft_frequencies, fft_arr[0])
    plt.show()

    #######################################################################################################

    min_y_axis, max_y_axis = 0, max(np.asarray(fft_arr).ravel())
    positive_fft_frequencies = np.round(np.asarray(positive_fft_frequencies), 2)
    
    def update(frame):
        plt.cla()  # Clear the current axes
        plt.plot(positive_fft_frequencies, fft_arr[frame], label=f'Filtered recieved signal', alpha=0.7)
        
        plt.legend(loc='upper right')
        plt.xlim(0, 80*1000)
        plt.ylim(min_y_axis, max_y_axis)
        plt.title(f'Pulse {frame}, starts at {times_reshaped[frame][0]} ms since hammer hit')
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Number of points')

    fig, ax = plt.subplots(figsize=(10, 5))
    ani = FuncAnimation(fig, update, frames=NUM_PULSES, repeat=False)

    # Save as GIF
    extension = ".mp4" if save_as_mp4 else ".gif"
    save_place = file_folder_name + "/" + specific_file_name + '_FFT_animation' + extension
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800) if save_as_mp4 else PillowWriter(fps=10)
    ani.save(save_place, writer)
    print(f"Saved at: {save_place}")
    plt.show()
    
def plot_2d(input_file_array, legends, use_integral = True, use_calculated_env = True, use_abs = True, folder="", title="fig", in_ms = True, comparing_filters = False):
    '''
    Plots a heatmap as a single line (the line being the intergal or average of each pulse.)
    Then, combines all trials in input_file_array into a single line.

    Inputs:
     - input_file_array: An array of input_files. Each input_file is the same format as input_files in plot_heat_map and get_reshaped_arrays.
     - legends: Titles of the plots to save
     - use_integral: True to use integral, false to use average. Doesn't really matter because both calculate the sum of the abs of each pulse anyways.
     - use_calculated_env: True to analyze the pulses from the calculated env, false to analyze the pulses from the circuit env
     - use_abs: True to nomalize to first pulse and sum the absolute differences of the pulses. False to just sum each pulse.
     - folder: Location to save images in. Do not end in "/"
     - title: Name of figure to save.

     Ouputs:
     - Lineplots for each file in input_file_array, saved to specified location.
    '''
    total_sum_signal_times = []
    total_sum_signal_voltages = []
    i = 0

    # To slice which part of the heatmap we are observing (set both to False to not slice)
    # We only need to slice on the box data.
    ignore_start_of_pulse = False
    start_border_ms = 0.3 # Only look at the part of the pulse after this time in ms in the pulse
    ignore_end_of_pulse = True
    end_border_ms = 1.2 # Only look at the part of the pulse before this time in ms in the pulse
    if (not in_ms):
        start_border_ms *= 0.001
        end_border_ms *= 0.001
        
    # Find y-limit
    abs_max = 0
    abs_min = 100000000

    # Loop over each trial to get the averages/variances
    trial_maximum_times = []
    for i in range(len(input_file_array)):
        input_files = input_file_array[i]
        hammer_times = input_files[0]
        hammer_recieved = input_files[1]
        emg_recieved = input_files[2]
        cuff_times_reshaped = input_files[3]
        cuff_recieved_reshaped = input_files[4]
        circuit_env_reshaped = input_files[5]
        calculated_envelope_reshaped = input_files[6]
        time_ticks = input_files[7]
        NUM_PULSES = input_files[8]

        # Getting the lines
        arr_of_interest = calculated_envelope_reshaped if use_calculated_env else circuit_env_reshaped
        these_times = []
        these_lines = []
        
            # Average each pulse
        for r in range(NUM_PULSES):

            # Get pulses
            pulse_of_interest = arr_of_interest[r]
            times_of_interest = cuff_times_reshaped[r]
            dt = np.mean(np.diff(times_of_interest))
            these_times.append(times_of_interest[0])

            # Slice off the beginning and end of the pulse, if desired
            first_pulse = arr_of_interest[0]
            start_border_index = 0
            if (ignore_start_of_pulse):
                start_border_index = np.searchsorted(times_of_interest, start_border_ms + times_of_interest[0], side='right')
                times_of_interest = times_of_interest[start_border_index:]
                pulse_of_interest = pulse_of_interest[start_border_index:]
                first_pulse = first_pulse[start_border_index:]
            end_border_index = 0
            if (ignore_end_of_pulse):
                end_border_index = np.searchsorted(times_of_interest, end_border_ms + times_of_interest[0], side='left')
                times_of_interest = times_of_interest[:end_border_index]
                pulse_of_interest = pulse_of_interest[:end_border_index]
                first_pulse = first_pulse[:end_border_index]

            if use_abs: # and (not (comparing_filters)): 
                pulse_of_interest = np.abs(pulse_of_interest - first_pulse)

            if use_integral: these_lines.append(np.sum(pulse_of_interest)*dt)
            else: these_lines.append(np.average(pulse_of_interest))

        # Finding y limits
        abs_max = max(abs_max, np.max(these_lines))
        abs_min = min(abs_min, np.min(these_lines))

        # Smoothen and find maximum
        start_border, end_border = 50, 200 # ms to look for maximum of contraction in
        if (not in_ms): 
            start_border *= 0.001
            end_border *= 0.001
        start_time_index = np.searchsorted(these_times, start_border, side='right')
        end_time_index = np.searchsorted(these_times, end_border, side='left')-1
        these_lines = gaussian_filter1d(these_lines, sigma=1)
        if (end_time_index < start_time_index):
            max_index = 0  # skip
        else:
            max_index = np.argmax(these_lines[start_time_index:end_time_index]) + start_time_index
            trial_maximum_times.append(these_times[max_index])

        # Plot each individual trial and its maximum
        plt.plot(these_times, these_lines, label = legends[i])
        plt.plot(these_times[max_index], these_lines[max_index], 'ro')
        plt.text(these_times[max_index], these_lines[max_index], f'Max {these_times[max_index]:.2f} ms,\n {these_lines[max_index]:.2f} mV', fontsize=6, ha='right', va='top')
        plt.xlabel("Time since hammer hit (ms)") if in_ms else plt.xlabel("Time since hammer hit (s)")
        plt.ylabel("Voltage (mV)")
        plt.title("Integral of each pulse over time")
        
        # Sum
        if len(total_sum_signal_voltages) == 0: 
            total_sum_signal_times = these_times
            total_sum_signal_voltages = these_lines
        else: 
            total_sum_signal_times, total_sum_signal_voltages = sum_signals(total_sum_signal_times, total_sum_signal_voltages, these_times, these_lines)
        i+=1

    # Save the image of all the overlayed trials.
    plt.legend()
    time_unit = "ms" if in_ms else "s"
    # variance_signal = np.round(np.var(total_sum_signal_voltages, axis=0), 2)
    variance_signal = np.round(np.std(trial_maximum_times), 2)
    save_path = folder + "/" + title + "_stddev_"+str(variance_signal)+time_unit+".png"
    plt.savefig(save_path)
    print("Saving to "+save_path)
    plt.ylim(abs_min, abs_max)
    plt.show()

    # For shifting the text of points so they don't overlap.
    x_offset = 1  # Adjust the horizontal shift (right)
    y_offset = (abs_max - abs_min)/10.0  # Adjust the vertical shift (up)

    # Filtering to smooth the signal
    total_sum_signal_voltages = gaussian_filter1d(total_sum_signal_voltages, sigma=1)/len(input_file_array)

    # Peaks and valleys of the "total"
    dt = np.mean(np.diff(total_sum_signal_times)) # in ms
    min_dist_btw_peaks_in_ms = 30
    peaks, _ = find_peaks(total_sum_signal_voltages, distance=min_dist_btw_peaks_in_ms/dt)
    valleys, _ = find_peaks(-1*total_sum_signal_voltages, distance=min_dist_btw_peaks_in_ms/dt)

    # Plot
    plt.plot(total_sum_signal_times, total_sum_signal_voltages)
    plt.ylim(abs_min, abs_max)
    plt.xlabel("Time since hammer hit (ms)") if in_ms else plt.xlabel("Time since hammer hit (s)")
    plt.ylabel("Voltage (mV)")
    plt.title("Integral of each pulse over time")

    # Mark and label peaks (maxima)
    plt.plot(total_sum_signal_times[peaks], total_sum_signal_voltages[peaks], 'ro', label='Maxima')
    for i in peaks:
        plt.text(total_sum_signal_times[i]+x_offset, total_sum_signal_voltages[i]+y_offset, f'Max {total_sum_signal_times[i]:.2f} '+time_unit+f',\n {total_sum_signal_voltages[i]:.2f} mV', fontsize=6, ha='right', va='top')

    # Mark and label valleys (minima)
    plt.plot(total_sum_signal_times[valleys], total_sum_signal_voltages[valleys], 'bo', label='Minima')
    for i in valleys:
        plt.text(total_sum_signal_times[i]-x_offset, total_sum_signal_voltages[i]-y_offset, f'Min {total_sum_signal_times[i]:.2f} '+time_unit+f',\n {total_sum_signal_voltages[i]:.2f} mV', fontsize=6, ha='left', va='bottom')
        
    save_path = folder + "/" + title + "_average.png"
    plt.savefig(save_path)
    print("Saving to "+save_path)
    plt.show()
        
########################################################################################################## Overall analsis of CSV functions

def analyze_one_trial(file_name, col_order, plot_circuit_envelope):
    '''
    For a single trial, saves the images of:
    
    IF LOOKING AT CIRCUIT ENVELOPE (plot_circuit_envelope = True)
    - the heat map (of circuit envelope height over time)
    - the 2D line plot (of average circuit envelope integral over time)

    IF LOOKING AT CALCULATED ENVELOPE (plot_circuit_envelope = False):
    - the heat map (of calculated envelope height over time, bandpassed 40kHz to 60kHz)
    - the heat map (of Fourier transform of the recieved signal over time)
    - The 2D line plot comparing 3 lines overlayed:
        - envelope of recieved pulses, unfiltered.
        - envelope of recieved pulses with a 40kHz to 60kHz filter on them ("high pass filter")
        - lowpass filter (how do the center of the pulses move up and down in time?) of the recieved pulses

    Inputs:
    - file_name: The path to the CSV or text file we wish to analyze
    - col_indexes: Tells us which column in your CSV corresponds to what data collected from oscilloscope. \n
        \t col_indexes = [times_col_index, hammer_index, transmit_col_index, recieve_col_index, circuit_col_index, emg_col_index = 1]
        (It tells us which column of the CSV corresponds to times, to hammer, to cuff, etc)
    - plot_circuit_envelope: If true, analyzes the measured circuit envelope, if false, analzes the calculated envelope of the data

    Outputs: Saves the images described above
    '''
    if (verbose): print("Starting analysis...")

    # 1. Replace "infinity" signs in oscilloscope file if present, and shape into numpy dataframe.
    very_negative_number = -100
    very_positive_number = 100
    file_folder_name = file_name[:file_name.rindex("/")]
    specific_file_name = file_name[file_name.rindex("/")+1:file_name.rindex(".")]
    my_csv = pd.read_csv(file_name,skiprows=1, sep=',' if file_name[-1]=="v" else "\s+")
    my_csv.replace([float('-inf'), '-∞'], very_negative_number, inplace=True)
    my_csv.replace([float('inf'), '∞'], very_positive_number, inplace=True)
    my_csv = my_csv.to_numpy()
    
    if (verbose): print(f"Finished reading in CSV {file_name}")

    #'''
    # Optional sanity-check plotting of the overall data.
    if (verbose):
        csv_times = my_csv[:,col_order[0]]
        csv_square_pulses = my_csv[:,col_order[2]]
        csv_recieved_pulses = my_csv[:,col_order[3]]
        plt.plot(csv_times, csv_square_pulses)
        plt.plot(csv_times, csv_recieved_pulses)
        plt.show()
        # '''
    
    # Some of the longer files will have the time unit as seconds, not ms, so check this.
    time_unit = check_time_unit(file_name)
    in_ms = time_unit == "(ms)"
    if (verbose): print(f"\tThe time unit of this CSV is {time_unit}")

    # 2. Get reshaped arrays. 
    if (verbose): print("\tGetting regular arrays (filtered 40 kHz to 60 kHz)...")
    regular_reshaped_arr = get_reshaped_arrays(my_csv, col_order, filter_freqs=[40000,60000], in_ms=in_ms)
        # used to get the unfiltered envelope
    if (verbose): print("\tGet unfiltered arrays and envelopes")
    unfiltered_reshaped_arr = get_reshaped_arrays(my_csv, col_order, filter_freqs=[20,1000000000], in_ms=in_ms)
        # used to get the lowpassed_signal
    if (verbose): print("\tGet lowpass array")
    low = get_reshaped_arrays(my_csv, col_order, filter_freqs=[1,10000], in_ms=in_ms, keep_raw_data=False)
        # set the so-called "calculated envelope" to actually just be the lowpassed signal reshaped
    lowpassed_reshaped_arr = [low[0], low[1], low[2], low[3], low[4], low[5], low[4], low[7], low[8]]

    # 3. Plots! These will save in the same folder that the file you are reading is in (unless you edit file_folder_name)
    if (verbose): print("Getting plots...")

    if (plot_circuit_envelope):
        # Heat map of circuit envelope
        plot_heat_map(regular_reshaped_arr, folder_path=file_folder_name, png_name=specific_file_name, stddev=3, plot_circuit_env=True, in_ms=in_ms)

        # 2D average map of circuit envelope.
        input_file_array = [regular_reshaped_arr]
        legends = [specific_file_name]
        plot_2d(input_file_array, legends, folder=file_folder_name, title=specific_file_name, use_calculated_env=False, in_ms=in_ms)

    else:
        # Heat map of filtered envelope.
        plot_heat_map(regular_reshaped_arr, folder_path=file_folder_name, png_name=specific_file_name, stddev=3, plot_circuit_env=False, in_ms=in_ms)
    
        # Heat map of Fourier Transform.
        # plot_heat_map_fourier(regular_reshaped_arr, folder_path=file_folder_name, png_name=specific_file_name, stddev=3, in_ms=in_ms)

        # 2D plot comparing 3 types of filtering
        legends = ["lowpassed", "bandpassed", "unfiltered"]
        input_file_array = [lowpassed_reshaped_arr, regular_reshaped_arr, unfiltered_reshaped_arr]
        plot_2d(input_file_array, legends, folder=file_folder_name, title=specific_file_name, use_calculated_env=True, comparing_filters = True)

    # GIF
    # get_fourier_gif(regular_reshaped_arr, col_order, save_as_mp4=False, file_folder_name=file_folder_name, specific_file_name=specific_file_name)

def summary_of_trials(file_names, col_order, experiment_name, analyze_circuit_env):
    '''
    For a single experiment, saves the images of:
    - the average heat map (of pulse envelope height over time, bandpassed 40kHz to 60kHz)
    - The 2D line plot (of pulse envelope height over time, bandpassed 40kHz to 60kHz)
        - One image showing each trial overlayed
        - One image showing the average of all the trials
    
    Inputs:
    - file_names: An array of the paths to the CSV or text files we wish to analyze
    - col_indexes: Tells us which column in your CSV corresponds to what data collected from oscilloscope. \n
        \t col_indexes = [times_col_index, hammer_index, transmit_col_index, recieve_col_index, circuit_col_index, emg_col_index = 1]
        (It tells us which column of the CSV corresponds to times, to hammer, to cuff, etc)
    - experiment_name: Used as part of the name of the images we save
    - analyze_circuit_env: If true, analyzes the measured circuit envelope, if false, analzes the calculated envelope of the data    
    
    Outputs: Saves the images described above
    '''

    legends = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]

    input_files_across_trials = []
    summed_heatmap_across_trials = []
    file_folder_name = ""

    # Loop through each file in the folder provided, extracting the heatmap array from each one.
    for file_name in file_names:
        # Get the name of file and folder the CSV is extracted from.
        file_folder_name = file_name[:file_name.rindex("/")]
        specific_file_name = file_name[file_name.rindex("/")+1:file_name.rindex(".")]

        # Some of the longer files will have the time unit as seconds, not ms, so check this.
        time_unit = check_time_unit(file_name)
        in_ms = time_unit == "(ms)"

        # Read the CSV and reshape the array.
        trial_csv = pd.read_csv(file_name,skiprows=1, sep=',' if file_name[-1]=="v" else "\s+")
        trial_csv.replace([float('-inf'), '-∞'], -100, inplace=True)
        trial_csv.replace([float('inf'), '∞'], 100, inplace=True)
        trial_csv = trial_csv.to_numpy()
        trial_arr = get_reshaped_arrays(trial_csv, col_order, filter_freqs=[40000,60000], in_ms=in_ms)
        reshaped_array_of_interest = trial_arr[5] if analyze_circuit_env else trial_arr[6]
        input_files_across_trials.append(trial_arr)

        # Running sum of heatmaps (the reshaped envelopes) across trials.
        # The slicing is to account for the fact that sometimes the length of the arrays is off by 1.
        if len(summed_heatmap_across_trials)==0: 
            summed_heatmap_across_trials = np.asarray(reshaped_array_of_interest)
        else: 
            smaller_row_length = min(len(summed_heatmap_across_trials), len(reshaped_array_of_interest)) 
            smaller_col_length = min(len(summed_heatmap_across_trials[0]), len(reshaped_array_of_interest[0])) 
            # print(f"HEATMAP ADDING: adding shapes {np.shape(summed_heatmap_across_trials)} to incoming shape {np.shape(reshaped_array_of_interest)}")
            summed_heatmap_across_trials = np.add(summed_heatmap_across_trials[0:smaller_row_length,0:smaller_col_length], 
                                   np.asarray(reshaped_array_of_interest)[0:smaller_row_length,0:smaller_col_length])

    # Formatting for plot_heat_map
    i = input_files_across_trials[0]    # Arbitrarily use the times and hammer strike from the first trial. 
    combined_input_file = []
    #        0             1               2                  3                     4                    5                       6                  7         8
    #  hammer_times, hammer_recieved, emg_recieved, cuff_times_reshaped, cuff_recieved_reshaped, circuit_env_reshaped, calculated_env_reshaped, time_ticks, NUM_PULSES
    if (analyze_circuit_env):
        combined_input_file = [i[0], i[1], i[2], i[3],i[4], summed_heatmap_across_trials/len(file_names), i[6],i[7], len(summed_heatmap_across_trials)]
    else: 
        combined_input_file = [i[0], i[1], i[2], i[3],i[4], i[5], summed_heatmap_across_trials/len(file_names), i[7], len(summed_heatmap_across_trials)]


    ############################################## ACTUAL PLOTTING #############################################    
    # Plot the overlayed line plots of each trial. The saved image will have the variance across lines in it.
    plot_2d(input_files_across_trials, legends, use_abs=True, use_calculated_env=(not analyze_circuit_env), folder=file_folder_name, title=experiment_name, in_ms = in_ms)

    # Plot the average heatmap across all trials for this experiment.
    plot_heat_map(combined_input_file, stddev=3, plot_circuit_env=analyze_circuit_env, folder_path=file_folder_name, png_name=experiment_name+"_average", in_ms = in_ms)

##########################################################################################################

if __name__ == "__main__":

    #####################################################################################################
    ################                     Commonly processed files                      ##################
    #####################################################################################################

    # SINA: 
    # T1-T5: B1.5 (lasts ~110 ms) – Reflex starts between 60 ms and 66.81 ms
    # T6-T10: B2 (lasts ~140 ms) – Reflex starts between 
    # T11: B4 (lasts ~290 ms)
    # T12-T17: B5.5 (lasts ~400 ms)
    # T18-T22: P1 (lasts ~470 ms)
    # T23-T27: P2 (lasts ~470 ms)
    # T28-32: P4 (lasts ~470 ms)
    # T33-37: P8 (lasts ~470 ms)

    pico_file_names_sina_P1 = [
        "src/app_v1/data_from_experiments/lower_resolution_longer_time_trials/sina/Pico/sina18.csv",
        "src/app_v1/data_from_experiments/lower_resolution_longer_time_trials/sina/Pico/sina19.csv",
        "src/app_v1/data_from_experiments/lower_resolution_longer_time_trials/sina/Pico/sina20.csv",
        "src/app_v1/data_from_experiments/lower_resolution_longer_time_trials/sina/Pico/sina21.csv",
        "src/app_v1/data_from_experiments/lower_resolution_longer_time_trials/sina/Pico/sina22.csv"
    ]

    pico_file_names_rachel_P1 = [
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/rachel/rachel_no_box/rachel_t1.csv",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/rachel/rachel_no_box/rachel_t2.csv",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/rachel/rachel_no_box/rachel_t3.csv",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/rachel/rachel_no_box/rachel_t4.csv",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/rachel/rachel_no_box/rachel_t5.csv",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/rachel/rachel_no_box/rachel_t6.csv",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/rachel/rachel_no_box/rachel_t7.csv",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/rachel/rachel_no_box/rachel_t8.csv",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/rachel/rachel_no_box/rachel_t9.csv",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/rachel/rachel_no_box/rachel_t10.csv"
    ]

    pico_file_names_priya_P1 = [
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/priya/priyatrial1.txt",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/priya/priyatrial2.txt",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/priya/priyatrial3.txt",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/priya/priyatrial4.txt",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/priya/priyatrial5.txt",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/priya/priyatrial6.txt",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/priya/priyatrial7.txt",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/priya/priyatrial8.txt",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/priya/priyatrial9.txt",
        "src/app_v1/data_from_experiments/reflex_by_subject/Pico/priya/priyatrial10.txt"
    ]
    #####################################################################################################
    #####################################################################################################
    #####################################################################################################

    # REMINDER: col_index_order: col_indexes = [times, hammer, transmit, recieved, circuit_env, emg = 1]
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Example usage: Uncomment to analyze one experiment.
    #''' 
    # !!!!!!!!!!!!!!! User should edit: file_names, col_order, experiment_name as needed.
    file_names = pico_file_names_rachel_P1
    col_order = [0, 3, 4, 1, 2, 1]  # time=col 0, hammer=col 3, transmitted=col 4, raw recieved=col 1, circuit env=col 2, emg=1 (dummy)
    experiment_name = "Exp_Summary_rachel_P1"
    analyze_circuit_envelope = False
    analyze_calculated_envelope = True
    # !!!!!!!!!!!!!!!

    #'''
    # Analyze each trial in the experiment.
    for file_name in file_names:
        if (analyze_circuit_envelope):
            analyze_one_trial(file_name, col_order, plot_circuit_envelope = True)
        if (analyze_calculated_envelope):
            analyze_one_trial(file_name, col_order, plot_circuit_envelope = False)
    #'''
    '''
    # Get a summary of the experiment.
    if (analyze_circuit_envelope):
        summary_of_trials(file_names, col_order, experiment_name, analyze_circuit_env = True)  # heat map, line plots of circuit env
    if (analyze_calculated_envelope):
        summary_of_trials(file_names, col_order, experiment_name, analyze_circuit_env = False)  # heat map, line plots of calculated env
    # '''  