import numpy as np
from glob import glob

def add_state_to_file(filename):
   temp = 534.67
   strain_rate = get_strain_rate(filename)
   oneline = f"{{'temperature':{temp}, 'engineering_strain_rate':{strain_rate} }}\n"

   with open(filename, 'r+') as fp:
       lines = fp.readlines()    # lines is list of line, each element '...\n'
       lines.insert(0, oneline)  # you can use any index if you know the line index
       fp.seek(0)                 # file pointer locates at the beginning to write the whole file again
       fp.writelines(lines)       # write whole lists again to the same file

def get_strain_rate(filename):
    if "0.01" in filename:
        strain_rate = 1e-2
    elif "0.0001" in filename:
        strain_rate = 1e-4
    elif "500" in filename:
        strain_rate = 500
    elif "1800" in filename:
        strain_rate = 1800
    elif "3600" in filename:
        strain_rate = 3600
    return strain_rate

filenames = glob("cleaned*.dat")
temp = 534.67
for filename in filenames:
    print(f"writing {filename}")

    data = np.genfromtxt(filename, skip_header=2, delimiter=',')

    if "sml" in filename.lower():
        header = "time, displacement, load, engineering_strain, engineering_stress"
    else:
        header = "time, transmitted displacement, incident displacement, load, engineering_strain, engineering_stress, displacement"
    new_filename = filename.split('dat')[0]+"csv"
    np.savetxt(new_filename, data, comments="", header=header, delimiter=',')
    add_state_to_file(new_filename)
    print(f"wrote {new_filename}")
