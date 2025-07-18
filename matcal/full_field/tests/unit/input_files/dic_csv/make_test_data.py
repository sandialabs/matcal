import numpy as np

x_range = np.linspace(0,1,40)
y_range = np.linspace(1,3,50)

x, y = np.meshgrid(x_range, y_range)
x = x.flatten()
y = y.flatten()
time = np.array([0,1,2])

def u_x(x,y,t):
  return x*.1*t

def u_y(x,y,t):
  return (2.-y)*.025*t

def temp(x,y,t):
  return 300*np.ones(np.shape(x))

base_name = "larger_dic_"
suffix = ".csv"
header = "X, Y, Ux, Uy, T"
for i in range(3):
  filename = "{}{}{}".format(base_name,i,suffix)
  t = time[i]
  np.savetxt(filename, np.array([x, y, u_x(x,y,t), u_y(x,y,t), temp(x,y,t)]).T, delimiter=',', header=header)
