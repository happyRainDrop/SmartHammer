
import matplotlib.pyplot as plt 
import csv 
import glob
import os
  
times = [] 
vals_1 = [] 

# read given file
list_of_files = glob.glob('C:/Users/tealw/Documents/PlatformIO/Projects/SmartHammer/src/logs/*txt')
latest_file = max(list_of_files, key=os.path.getctime)
file_name = latest_file
print("reading "+file_name)

with open(file_name,'r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    time = 0
    time_interval = 4; # in microseconds
    i = -1
    for row in lines: 
        i += 1
        if (i==0): continue
        times.append(float(row[0])) # convert to millesecond 
        vals_1.append(float(row[1]))

plt.plot(times, vals_1, color = "gray") 
plt.xticks(rotation = 25) 
plt.xlabel('Time (milliseconds)') 
plt.ylabel('Voltage (V)') 
plt.title('Signal read from function generator', fontsize = 20) 
  
plt.show() 