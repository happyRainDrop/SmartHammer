
import matplotlib.pyplot as plt 
import csv 
import glob
import os
  
times1 = [] 
times2 = [] 
vals_1 = [] 
vals_2 = [] 
pot_val = 1000
ref_voltage = 5
MAX_TIME = 10 # in ms

''' FIRST, READ HAMMER '''
def raw_val_to_force(raw_val):
    m = 0.0151105
    b= -0.29373
    # y = mx + b: force (lbs) = m*(conductance (uS)) + b
    # best fit from datasheet: -0.29373+0.0151105x

    read_voltage = raw_val * ref_voltage / 1023.0
    conductance = read_voltage / (pot_val*(ref_voltage-read_voltage))
    force_lbs = m * (conductance*1000000) + b
    return read_voltage

# read latest file
list_of_files = glob.glob('C:/Users/tealw/Documents/PlatformIO/Projects/SmartHammer/src/logs/*txt')
latest_file = max(list_of_files, key=os.path.getctime)
file_name = latest_file
print("reading "+file_name)

with open(file_name,'r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    start_time = 0;
    for row in lines: 
        if (len(row)>1 and int(row[1])==0):
            continue

        if (start_time == 0): 
            start_time = float(row[0])

        time = (float(row[0])-start_time)/1000.0
        if (time > MAX_TIME):
            break

        times1.append(time) # convert to millesecond 
        vals_1.append(raw_val_to_force(int(row[1])))

''' THEN, READ TRIGGERED DUO '''
# read given file
file_name = 'C:/Users/tealw/Documents/PlatformIO/Projects/SmartHammer/src/logs/output.csv'
print("reading "+file_name)

with open(file_name,'r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    time = 0
    time_interval = 4; # in microseconds
    i = -1
    for row in lines: 
        i+=1
        if (i<=0): continue

        if (float(row[0]) > MAX_TIME):
            break

        times2.append(float(row[0])) # convert to millesecond 
        vals_2.append(float(row[1]) + 0.025)


plt.plot(times1, vals_1, color = "blue", alpha=0.5)
plt.plot(times2, vals_2, color = "red", alpha = 0.5) 

plt.xticks(rotation = 25) 
plt.xlabel('Time (milleseconds)') 
plt.ylabel('Voltage (V)') 
plt.title('Force over time', fontsize = 20) 
  
plt.show() 