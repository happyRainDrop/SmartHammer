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

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_filtered_pulses(data, Sampling_frequency, use_raw_envelope = False):
    if (use_raw_envelope):
        kernel_size = 3
        return medfilt(data, kernel_size)
    else:
        Filter_lowcut =40000
        Filter_highcut =60000
        Filter_order = 4
        # print(f"Sampling frequency: {Sampling_frequency} Hz")
        data = np.asarray(data, dtype=float)
        sos = butter(Filter_order, [Filter_lowcut, Filter_highcut], btype='bandpass', fs=Sampling_frequency, output='sos')
        return np.apply_along_axis(lambda x: sosfilt(sos, x), axis=0, arr = data)

def get_envelope(data, use_raw_envelope = False):
    if use_raw_envelope:
        lowest_freq_of_recieved_sig = 20000 # needs to be much lower than 52 kHz -- this is what determines the distance between points on the envelope
        osc_sampling_freq = 200000
        calculated_envelope = find_maxima_envelope(data, osc_sampling_freq, lowest_freq_of_recieved_sig)
    else:
        calculated_envelope = np.abs(hilbert(data, axis=0))

    return calculated_envelope

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

####################################################################################################### Processing functions

def get_reshaped_arrays(all_csv_data, col_indexes):
    '''
    Input:. \n
        all_csv_data: your csv data as a pandas dataframe
        col_indexes: Tells us which column in your CSV corresponds to what data collected from oscilloscope!
        col_indexes = [times_col_index, hammer_index, transmit_col_index, recieve_col_index, circuit_col_index, emg_col_index = 1]
    Return: hammer_times, hammer_recieved, emg_recieved, cuff_times_reshaped, cuff_recieved_reshaped, circuit_env_reshaped, time_ticks, NUM_PULSES
    '''
    use_raw_envelope = False
    keep_raw_data = True

    csv_times = all_csv_data[:,col_indexes[0]]
    csv_square_pulses = all_csv_data[:,col_indexes[2]]
    csv_recieved_pulses = all_csv_data[:,col_indexes[3]]
    csv_circuit_envelope =  all_csv_data[:,col_indexes[4]]
    csv_hammer = all_csv_data[:,col_indexes[1]]
    emg_recieved = all_csv_data[:, col_indexes[5]]

    dt = np.mean(np.diff(csv_times))
    fs = 1000 / dt  # Sampling frequency
    ############################################

    hammer_times, hammer_recieved = csv_times, csv_hammer

    filtered_recieved_pulses = get_filtered_pulses(csv_recieved_pulses, fs, use_raw_envelope)
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
            i += 500
        i+=1
    # print(max_pulse_length_in_indicies)
    NUM_PULSES = r

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
    fs = 1000 / dt  # Sampling frequency

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

def plot_heat_map(input_files, folder_path = "", png_name = "", stddev = 3, use_emg = False, plot_circuit_env = False):
    
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
    ax2 = plt.subplot(gs[1])
    cuff_vals_for_heatmap = np.asarray(calculated_envelope_reshaped) - np.asarray(calculated_envelope_reshaped)[0, :]
    if plot_circuit_env: 
        cuff_vals_for_heatmap = np.asarray(circuit_env_reshaped) - np.asarray(circuit_env_reshaped)[0, :]
    lower_outliers, upper_outliers, lower_lim_imshow, upper_lim_imshow = find_outliers_std(cuff_vals_for_heatmap, stddev)
    im = ax2.imshow(np.transpose(cuff_vals_for_heatmap), aspect='auto', cmap='jet', vmin=lower_lim_imshow, vmax=upper_lim_imshow)
    ax2.set_title('Circuit envelope: \nPulse height vs time normalize to start of pulse, all pulses overlayed')
    ax2.set_ylabel('Array index within pulse')
    ax2.set_xlabel('Start time of pulse (ms)')
    time_tick_positions = np.arange(0, NUM_PULSES, NUM_PULSES / len(time_ticks))
    ax2.set_xticks(ticks=time_tick_positions[0::int(len(time_ticks)/10)])
    ax2.set_xticklabels(labels=time_ticks[0::int(len(time_ticks)/10)])
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
    str_name = folder_path + "/" + png_name + '.png'
    if len(str_name) > 5: fig.savefig(str_name)
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

def get_gif(all_csv_data, col_indexes, save_as_mp4 = True, plot_hammer = False, plot_circuit_envelope = True, plot_calculated_envelope = True, compare_contraction = False, active_recieved_pulses_filtered = None, file_folder_name = "", specific_file_name = ""):
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

    lower_outliers, upper_outliers, min_y_axis, max_y_axis = find_outliers_std(np.asarray(recieved_pulses_reshaped))
    scale_circuit_envelope = max_y_axis*1.0/max(csv_circuit_envelope)
    scale_hammer = max_y_axis*1.0/max(csv_hammer)

    shift_active_up =  np.mean(np.asarray(recieved_pulses_reshaped)) - np.mean(active_recieved_pulses_filtered) if compare_contraction else 0
    shift_calc_env_up = 10 + np.mean(np.asarray(recieved_pulses_reshaped)) - np.mean(np.asarray(calculated_envelope_reshaped))

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

def plot_2d(input_file_array, legends, use_integral = True, use_calculated_env = True, folder="", title="fig"):
    times_arr = []
    lines_arr = []
    i = 0
    
    for input_files in input_file_array:
        hammer_times = input_files[0]
        hammer_recieved = input_files[1]
        emg_recieved = input_files[2]
        cuff_times_reshaped = input_files[3]
        cuff_recieved_reshaped = input_files[4]
        circuit_env_reshaped = input_files[5]
        calculated_envelope_reshaped = input_files[6]
        time_ticks = input_files[7]
        NUM_PULSES = input_files[8]

        dt = np.mean(np.diff(hammer_times))

        arr_of_interest = calculated_envelope_reshaped if use_calculated_env else circuit_env_reshaped
        these_times = []
        these_lines = []
        for r in range(NUM_PULSES):
            these_times.append(cuff_times_reshaped[r,0])
            if use_integral: these_lines.append(np.sum(arr_of_interest[r])*dt)
            else: these_lines.append(np.average(arr_of_interest[r]))
        
        times_arr.append(np.asarray(these_times))
        plt.plot(these_times, these_lines, label = legends[i])
        plt.ylim(0, 50)
        plt.xlabel("Time since hammer hit (ms)")
        plt.ylabel("Voltage (mV)")
        plt.title("Integral of each pulse over time")

        if len(lines_arr) == 0: lines_arr = np.asarray(these_lines)
        else: 
            lines_arr = np.add(np.asarray(these_lines[:min(len(these_lines), len(lines_arr))]), 
                                 np.asarray(lines_arr[:min(len(these_lines), len(lines_arr))]))
        i+=1
    
    variance_signal = np.round(np.var(lines_arr, axis=0), 2)
    save_path = folder + "/" + title + "_individual_variance_"+str(variance_signal)+".png"
    plt.savefig(save_path)
    print("Saving to "+save_path)
    plt.ylim(20, 40)
    plt.show()

    # To account for different lengths
    times_cut = times_arr[0][:min(len(times_arr[0]), len(lines_arr))]
    lines_cut =  (lines_arr/(1.0*len(times_arr)))[:min(len(times_arr[0]), len(lines_arr))] 
    lines_cut = gaussian_filter1d(lines_cut, sigma=1)
    peaks, _ = find_peaks(lines_cut, distance=20)
    valleys, _ = find_peaks(-1*lines_cut, distance=20)

    # Plot
    plt.plot(times_cut, lines_cut)
    plt.ylim(20, 40)
    plt.xlabel("Time since hammer hit (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Integral of each pulse over time")

    # Mark and label peaks (maxima)
    plt.plot(times_cut[peaks], lines_cut[peaks], 'ro', label='Maxima')
    for i in peaks:
        plt.text(times_cut[i], lines_cut[i], f'Max {times_cut[i]:.2f} ms,\n {lines_cut[i]:.2f} mV', fontsize=6, ha='right', va='top')

    # Mark and label valleys (minima)
    plt.plot(times_cut[valleys], lines_cut[valleys], 'bo', label='Minima')
    for i in valleys:
        plt.text(times_cut[i], lines_cut[i], f'Min {times_cut[i]:.2f} ms,\n {lines_cut[i]:.2f} mV', fontsize=6, ha='left', va='bottom')
        
    save_path = folder + "/" + title + "_average.png"
    plt.savefig(save_path)
    print("Saving to "+save_path)
    plt.show()
        


if __name__ == "__main__":
    # col_indexes = [times, hammer, transmit, recieved, circuit_env, emg = 1]
    very_negative_number = -100
    very_positive_number = 100

    file_name = "src/app_v1/data_from_experiments/reflex_by_subject/Pico/rachel/rachel_10.csv"
    file_folder_name = file_name[:file_name.rindex("/")]
    specific_file_name = file_name[file_name.rindex("/")+1:file_name.rindex(".")]
    col_order = [0, 3, 4, 1, 2, 1]  # time, recieved, env, hammer, square

    my_csv = pd.read_csv(file_name,skiprows=1, sep=',' if file_name[-1]=="v" else "\s+")
    my_csv.replace([float('-inf'), '-∞'], very_negative_number, inplace=True)
    my_csv.replace([float('inf'), '∞'], very_positive_number, inplace=True)
    my_csv = my_csv.to_numpy()
    my_arr = get_reshaped_arrays(my_csv, col_order)
   
    # plot_heat_map(my_arr, folder_path=file_folder_name, png_name=specific_file_name, stddev=3, plot_circuit_env=True)
    get_gif(my_csv, col_order, plot_circuit_envelope = True, file_folder_name=file_folder_name, specific_file_name=specific_file_name)
    
    '''
    The below is for analyzing multiple trials at once.
    
    legends = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]
    col_order = [0, 2, 3, 1, 1, 1]  # time, recieved, hammer, square

    file_names = ["src/app_v1/data_from_experiments/reflex_by_subject/Pico/sophie/sophietrial1.txt", 
                  "src/app_v1/data_from_experiments/reflex_by_subject/Pico/sophie/sophietrial2.txt",
                  "src/app_v1/data_from_experiments/reflex_by_subject/Pico/sophie/sophietrial3.txt",
                  "src/app_v1/data_from_experiments/reflex_by_subject/Pico/sophie/sophietrial4.txt",
                  "src/app_v1/data_from_experiments/reflex_by_subject/Pico/sophie/sophietrial5.txt",
                  "src/app_v1/data_from_experiments/reflex_by_subject/Pico/sophie/sophietrial6.txt", 
                  "src/app_v1/data_from_experiments/reflex_by_subject/Pico/sophie/sophietrial7.txt",
                  "src/app_v1/data_from_experiments/reflex_by_subject/Pico/sophie/sophietrial8.txt",
                  "src/app_v1/data_from_experiments/reflex_by_subject/Pico/sophie/sophietrial9.txt",
                  "src/app_v1/data_from_experiments/reflex_by_subject/Pico/sophie/sophietrial10.txt"]

    input_files_across_trials = []
    total_thing = []
    file_folder_name = ""
    this_file_name = "sophie"
    for file_name in file_names:
        file_folder_name = file_name[:file_name.rindex("/")]
        specific_file_name = file_name[file_name.rindex("/")+1:file_name.rindex(".")]
        t1_csv = pd.read_csv(file_name,skiprows=1, sep=',' if file_name[-1]=="v" else "\s+").to_numpy()
        t1_arr = get_reshaped_arrays(t1_csv, col_order)
        if len(total_thing)==0: total_thing = np.asarray(t1_arr[5])
        else: total_thing = np.add(total_thing[:min(len(total_thing), len(t1_arr[5]))], 
                                   np.asarray(t1_arr[5][:min(len(total_thing), len(t1_arr[5]))]))

        input_files_across_trials.append(t1_arr)
    print(file_folder_name)
    # plot_2d(input_files_across_trials, legends, folder=file_folder_name, title=this_file_name)

    i = input_files_across_trials[0]
    plot_heat_map([i[0], i[1], i[2], i[3], i[4], total_thing/len(file_names), i[6], i[7], i[8]])
    '''  