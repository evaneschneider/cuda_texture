import matplotlib.pyplot as plt
import numpy as np

T = np.linspace(1.0, 9.0, 1000)

x_i = []
y_i = []
z_i = []
f = open('cloudy_coolingcurve_n3_HM05.txt', 'r')
f.readline()
lines = f.readlines()
f.close()
for line in lines:
  p = line.split()
  x_i.append(float(p[1]))
  y_i.append(float(p[2]))
  z_i.append(float(p[3]))
x = np.array(x_i)
y = np.array(y_i)
z = np.array(z_i)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
line1, = ax1.plot(x, y, 'bo', label="Cloudy data")
line2, = ax1.plot(x, z, 'ro', label="Cloudy data")
plt.xlabel("log(T) [K]");
plt.ylabel("log($\Lambda$ / n$_{h}$$^{2}$) [erg cm$^{-3}$ s$^{-1}$]");


x_i = []
y_i = []
f = open('cuda_coolingcurve_n3.txt', 'r')
lines = f.readlines()
f.close()
for line in lines:
  p = line.split()
  x_i.append(float(p[0]))
  y_i.append(float(p[1]))
x = np.array(x_i)
y = np.array(y_i)

line3, = ax1.plot(T, x, color="green", label="2D interpolation")
line4, = ax1.plot(T, y, color="green")


plt.legend(handles=[line1, line3], loc=2)
plt.show()
fig.savefig('cuda_coolingcurve_n3.png')
