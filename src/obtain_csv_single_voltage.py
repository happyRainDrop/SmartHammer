"""
App to read in varying voltages from Arduino.

To serve the app, run

    bokeh serve --show "C:/Users/tealw/Documents/PlatformIO/Projects/SmartHammer/src/obtain_csv.py"

on the command line.
"""

import asyncio
import re
import time
import signal
import csv
import sys

import serial
import serial.tools.list_ports

file_path_prefix = "C:/Users/tealw/Documents/PlatformIO/Projects/SmartHammer/src/logs/"
arduino_delim = "====="  # We print this between readings.

########################################### ARDUINO CONNECTIVITY FUNCTIONS #######################
def find_arduino(port=None):
    """Get the name of the port that is connected to Arduino."""
    if port is None:
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if p.manufacturer is not None and "Arduino" in p.manufacturer:
                port = p.device
                return port
    return port


def handshake_arduino(arduino, sleep_time=1, print_handshake_message=False, handshake_code=0):
    """Make sure connection is established by sending and receiving bytes."""
    # Close and reopen
    arduino.close()
    arduino.open()

    # Chill out while everything gets set
    time.sleep(sleep_time)

    # Set a long timeout to complete handshake
    timeout = arduino.timeout
    arduino.timeout = 2

    # Read and discard everything that may be in the input buffer
    _ = arduino.read_all()

    # Send request to Arduino
    arduino.write(bytes([handshake_code]))

    # Read in what Arduino sent
    handshake_message = arduino.read_until()

    # Send and receive request again
    arduino.write(bytes([handshake_code]))
    handshake_message = arduino.read_until()

    # Print the handshake message, if desired
    if print_handshake_message:
        print("Handshake message: " + handshake_message.decode())

    # Reset the timeout
    arduino.timeout = timeout

################################################## READ FROM SERIAL PORT FUNCTIONS #################

def read_all(ser, read_buffer=b"", **args):
    """Read all available bytes from the serial port and append to the read buffer.

    Parameters
    ----------
    ser : serial.Serial() instance
        The device we are reading from.
    read_buffer : bytes, default b''
        Previous read buffer that is appended to.

    Returns
    -------
    output : bytes
        Bytes object that contains read_buffer + read.

    Notes
    -----
    .. `**args` appears, but is never used. This is for compatibility with `read_all_newlines()` as a drop-in replacement for this function.
    """
    # Set timeout to None to make sure we read all bytes
    previous_timeout = ser.timeout
    ser.timeout = None

    in_waiting = ser.in_waiting
    read = ser.read(size=in_waiting)

    # Reset to previous timeout
    ser.timeout = previous_timeout

    return read_buffer + read


def read_until_delimiter(ser, delimiter=arduino_delim, read_buffer=b""):
    """Read data in until encountering the delimiter.

    Parameters
    ----------
    ser : serial.Serial() instance
        The device we are reading from.
    delimiter : str, default "====="
        The delimiter to read until.
    read_buffer : bytes, default b''
        Previous read buffer that is appended to.

    Returns
    -------
    output : bytes
        Bytes object that contains read_buffer + read.

    Notes
    -----
    .. This is a drop-in replacement for read_all_newlines().
    """
    raw = read_buffer
    delimiter_bytes = delimiter.encode()

    while delimiter_bytes not in raw:
        raw += ser.read(1)

    return raw


def parse_read(read):
    """Parse a read with time, voltage data

    Parameters
    ----------
    read : byte string
        Byte string with comma delimited time/voltage measurements.

    Returns
    -------
    time_ms : list of ints
        Time points in milliseconds.
    voltage : list of floats
        Voltages in volts.
    remaining_bytes : byte string
        Remaining, unparsed bytes.
    """
    time_ms = []
    voltage = []

    # Separate independent time/voltage measurements
    pattern = re.compile(rb"\d+\.\d+,\d+\.\d+")
    raw_list = [b"".join(pattern.findall(raw)).decode() for raw in read.split(b"\r\n")]

    for raw in raw_list[:-1]:
        try:
            t, V = raw.split(",")
            time_ms.append(float(t))
            voltage.append(float(V))
        except:
            pass

    if len(raw_list) == 0:
        return time_ms, voltage, b""
    else:
        return time_ms, voltage, raw_list[-1].encode()

################################################## FILL ARRAY AND SAVE TO CSV FUNCTIONS ##################

async def read_arduino_serial(arduino, times, voltages, delay=1, reader=read_until_delimiter):
    """Obtain streaming data"""
    # Specify delay (in ms)
    arduino.write(bytes([READ_DAQ_DELAY]) + (str(delay) + "x").encode())

    # Start streaming
    arduino.write(bytes([STREAM]))

    # Receive data
    read_buffer = [b""]
    while True:
        start_time = asyncio.get_event_loop().time()
        print("starting...")

        # Read in chunk of data
        raw = reader(arduino, read_buffer=read_buffer[0])

        # Parse it, passing if it is gibberish
        try:
            t, V, read_buffer[0] = parse_read(raw)

            # Update data dictionary
            times += t
            voltages += V
            
            # complete (only read one chunk)
            arduino.write(bytes([ON_REQUEST]))
            print(f"Time elapsed: {(asyncio.get_event_loop().time() - start_time)} s")
            save_to_csv(times, voltages)
            sys.exit(0)

        except Exception as e:
            print(f"Error parsing read: {e}")

        # Calculate elapsed time
        elapsed_time = (asyncio.get_event_loop().time() - start_time) * 1000

        # Calculate sleep time
        sleep_time = delay - elapsed_time
        if sleep_time > 0:
            await asyncio.sleep(sleep_time / 1000)

def save_to_csv(time_list, value_list, filename=file_path_prefix + 'output.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time (ms)', 'Voltage (V)'])  # Header row
        for time, value in zip(time_list, value_list):
            writer.writerow([time, value])
    print(f"Data saved to {filename}")

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    save_to_csv(times, voltages)
    sys.exit(0)

################################################## SET UP CONNECTION AND READ!! ##################

# Set up connection
HANDSHAKE = 0
VOLTAGE_REQUEST = 1
ON_REQUEST = 2
STREAM = 3
READ_DAQ_DELAY = 4

# Set up data dictionaries
times = []
voltages = []

async def main():
    # Initialize your Arduino connection here
    port = find_arduino()
    print(f"Found Arduino on port {port}.")
    arduino = serial.Serial(port, baudrate=115200)
    handshake_arduino(arduino, 1, True)
    await read_arduino_serial(arduino, times, voltages)

# Set the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)
# print('Press Ctrl+C to exit and save data.')

####################################################################################################

try:
    asyncio.run(main())  # Collect voltages.
except KeyboardInterrupt:
    # If signal handler is set up correctly, this should not be needed.
    pass
