import serial
import csv
import re
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# Flag to signal the thread to stop
stop_thread = threading.Event()
a = 1

# Function to read from the serial port and save to a CSV file
def read_from_serial(port, baudrate, output_file, data_queue):
    ser = serial.Serial(port, baudrate)

    while not stop_thread.is_set():
        try:
            line = ser.readline()
            try: 
                line = line.decode('utf-8').strip()
                if re.match(r'^-?\d+(\.\d+)?, -?\d+(\.\d+)?$', line):
                    time, voltage = line.split(',')
                    data_queue.append((float(time), float(voltage)))
                    if len(data_queue) > 100:
                        data_queue.popleft()  # Keep the length of the queue manageable
                    # print(f"Time: {time}, Voltage: {voltage}")
                else:
                    a = 1
                    # print(f"Format of {line} is wrong.")
            except:
                print(f"{port}: line {line} invalid.")
        except serial.SerialException:
            print("Serial exception occurred. Stopping data capture.")
            break
        except KeyboardInterrupt:
            print("Stopping data capture due to keyboard interrupt.")
            break
        except Exception as e:
            print(f"Unexpected error: {e}. Stopping data capture.")
            break

    print(f"Time to stop. Closing {port}")
    ser.close()
    return

# Function to start reading from serial in a separate thread
def start_serial_reading(port, baudrate, output_file, data_queue):
    thread = threading.Thread(target=read_from_serial, args=(port, baudrate, output_file, data_queue))
    thread.start()
    return thread

# Function to animate the plot
def animate(i, line, data_queue):
    if data_queue:
        times, voltages = zip(*data_queue)
        line.set_data(times, voltages)

# Function to handle the close event
def on_close(event):
    stop_thread.set()
    print("Plot window closed. Stopping data capture.")

# Example usage
if __name__ == "__main__":
    port1 = "COM20"  # Replace with your Arduino's serial port
    baudrate1 = 115200
    output_file1 = None
    
    data_queue1 = deque(maxlen=100)  # Queue to store incoming data for live plotting

    thread1 = start_serial_reading(port1, baudrate1, output_file1, data_queue1)

    # Set up the plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Live Data from Cuff')
    ax.set_ylim(bottom=0, top=3.3)
    ax.set_xlim(left=-0.1, right=0.9)

    ani = animation.FuncAnimation(fig, animate, fargs=(line, data_queue1), interval=0)
    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()

    # Wait for the thread to finish
    stop_thread.set()  # Ensure the thread is signaled to stop
    thread1.join(timeout=5)  # Wait for the thread to finish with a timeout

    print("Calibration complete.")
