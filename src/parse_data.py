
import matplotlib.pyplot as plt 
import csv 
import glob
import os
  
times = [] 
vals_1 = [] 
vals_2 = [] 
vals_3 = [] 
vals_4 = [] 
pot_val = 1000
ref_voltage = 3.3
MAX_TIME = 10 # in ms

def raw_val_to_force(raw_val):
    m = 0.0151105
    b= -0.29373
    # y = mx + b: force (lbs) = m*(conductance (uS)) + b
    # best fit from datasheet: -0.29373+0.0151105x

    read_voltage = raw_val * ref_voltage / 1024.0
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

        times.append(time) # convert to millesecond 
        vals_1.append(raw_val_to_force(int(row[1]))) 
        vals_2.append(raw_val_to_force(int(row[2]))) 
        vals_3.append(raw_val_to_force(int(row[3]))) 
        vals_4.append(raw_val_to_force(int(row[4]))) 

plt.plot(times, vals_1, color = "gray")
plt.plot(times, vals_2, color = "yellow") 
plt.plot(times, vals_3, color = "green") 
plt.plot(times, vals_4, color = "blue") 

plt.xticks(rotation = 25) 
plt.xlabel('Time (milleseconds)') 
plt.ylabel('Force (lbs)') 
plt.title('Force over time', fontsize = 20) 
  
plt.show() 