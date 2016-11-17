import matplotlib.pyplot as plt
import numpy as np
import sys

write = sys.stdout.write
def md_print(x): #print nparray as tex_array
    n = len(x)
    for i in range(0, n):
        m = len(x[i])
        for j in range(0, m - 1):
            write(str(x[i][j])+' & ')
        print x[i][m-1], '\\\\'

def cr(n):
    data = np.reshape(np.random.randint(0, 10, n*2), [2, n])
    return data

def cr_quad(n, par, sigma):
    data = np.zeros((2, n))
    data[0, :] = np.random.randint(0, 10, n)
    data[1, :] = par[0] + par[1]*data[0, :] + par[2]*data[0, :]**2 + (1. - 2*np.random.rand(n))*sigma
    return data

def lin(data):
    m = np.ones(data.shape).transpose()
    m[:,1] = data[0,:]
    ans = np.linalg.pinv(m.transpose().dot(m)).dot(m.transpose()).dot(data[1,:])
    return ans

def quad(data):
    m = np.ones((3, data.shape[1])).transpose()
    m[:,1] = data[0,:]
    m[:,2] = data[0,:]**2
    ans = np.linalg.pinv(m.transpose().dot(m)).dot(m.transpose()).dot(data[1,:])
    return ans

def get_line(data):
    x = [0, 10]
    y = [0]*2
    for i in [0, 1]:
        y[i] = data[0] + x[i]*data[1]
    return [x, y]

def get_quad(data):
    x = np.arange(0, 10, 0.1)
    y = np.zeros(x.shape)
    y = data[0] + x*data[1] + data[2]*x**2
    return [x, y]

fig = plt.figure()
plt.hold(True)




params = 10*(0.5-np.random.rand(3))
print 'Params = {}'.format(params)

data = np.array([[8., 6., 9., 4., 0.], # current data
 [237.05367751, 136.97838057, 297.16573452, 64.92781108, 3.65281937]])
n = 5
sigma = 0.5
# data =  cr_quad(n, params, sigma)

print data
opt_line = lin(data)
print "Optimal line is {}".format(opt_line)

opt_quad = quad(data)
print "Optimal quad is {}".format(opt_quad)

plt.scatter(*data, color='red', linewidths=3)

plt.plot(*get_line(opt_line))

plt.plot(*get_quad(opt_quad))

plt.grid(True)

plt.hold(False)
plt.show()
