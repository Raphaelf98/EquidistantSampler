#!/usr/local/opt/python/libexec/bin/python
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from math import cos, sin, sqrt, pi
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.misc import derivative

plt.rcParams['legend.fontsize']=10
fig= plt.figure()
ax = fig.gca(projection='3d')
#Polynomial function definition
px = np.poly1d([1,1,0])
py = np.poly1d([2,1,2,0])
pz = np.poly1d([2,1,0])
xnp = np.poly1d(px )
ynp = np.poly1d(py)
znp = np.poly1d(pz)

x = lambda t : xnp(t)
y = lambda t : ynp(t)
z = lambda t : znp(t)

#total arc length calculation at interval [min,max]
def arclength1(min,max):
    solutions = np.array([0,0])
    dx = lambda t: derivative(x, t, dx=1e-8)
    dy = lambda t:  derivative(y, t, dx=1e-8)
    dz = lambda t: derivative(z, t, dx=1e-8)
    solutions= quad(lambda t:np.sqrt(dx(t)**2+dy(t)**2+dz(t)**2), min , max)

    return solutions[0]

def integrand(t):
    dx = lambda t: derivative(x, t, dx=1e-8)
    dy = lambda t:  derivative(y, t, dx=1e-8)
    dz = lambda t: derivative(z, t, dx=1e-8)

    return sqrt(dx(t)**2+dy(t)**2+ dz(t)**2)

def curve_length(t0, S, length,quadtol):
    integral = quad(S, 0, t0,epsabs=quadtol,epsrel=quadtol)

    return integral[0] - length
#numeric solver: for given arc length determine function parameter t
def solve_t(curve_diff, length,opttol=1.e-15,quadtol=1e-10):
    return fsolve(curve_length, 0.0, (curve_diff, length,quadtol), xtol = opttol)[0]

def equidistantSampler(min, max, samples):
    equidistantSet =[]
    arcLength = arclength1(min,max)
    print('arclength',arcLength )
    sampleLength = arcLength /(samples-1)
    samplePos = 0
    for i in range(0,samples):
        equidistantSet.append(round(solve_t(integrand, samplePos,opttol=1e-5,quadtol=1e-3),6))
        samplePos = samplePos + sampleLength
    print('Equidistant set: ',equidistantSet)

    return np.array(equidistantSet)


t = np.linspace(0,1, 1000)
ax.plot(x(t),y(t),z(t), label= 'Parametric curve')

t_ = np.linspace(0,1, 24, endpoint= True)
ax.plot(x(t_),y(t_),z(t_), 'k+', label= 'Paramter t divided')
sampleSize= 25
te = equidistantSampler(0,1,sampleSize)
ax.plot(x(te),y(te),z(te),'ro', markersize=2, label= 'Equidistant points')
ax.plot([],[],[],' ', markersize=2, label= 'Sample size: '+ str(sampleSize))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()
plt.show()
