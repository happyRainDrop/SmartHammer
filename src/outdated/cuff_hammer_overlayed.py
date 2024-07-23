
import matplotlib.pyplot as plt 
import csv 
import glob
import os
  
times_cuff = [] 
vals_cuff = [] 
times_hammer = []
vals_hammer = []

# read given file
file_name_cuff = "C:/Users/tealw/Documents/PlatformIO/Projects/SmartHammer/src/logs/cuff_data.txt"
file_name_hammer = "C:/Users/tealw/Documents/PlatformIO/Projects/SmartHammer/src/logs/hammer_data.txt"

with open(file_name_cuff,'r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    for row in lines: 
        times_cuff.append(float(row[0])) # in milleseconds since hammer hit 
        vals_cuff.append(float(row[1]))

with open(file_name_hammer,'r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    for row in lines: 
        times_hammer.append(float(row[0])) # in milleseconds since hammer hit 
        vals_hammer.append(float(row[1]))

plt.plot(times_cuff, vals_cuff, color = "gray", label="Cuff recieved signal envelope") 
plt.plot(times_hammer, vals_hammer, color = "blue", label="Hammer force") 
plt.xticks(rotation = 25) 
plt.xlabel('Time (milliseconds)') 
plt.ylabel('Voltage (V)') 
plt.title('Signal read from cuff and hammer since hammer striked', fontsize = 20) 
plt.legend(loc="upper left")  
plt.show() 