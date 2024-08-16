import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np

### Written by Berjer Ding for the Magnetostatics unit in PHY250
### Plots the undimensionalized trajectory of a particle in an magnetic field given by the command-line.
### Write the electric field and initial coordinates in Cartesian coordinates with brackets and comma to separate coordinates, i.e. (a,b,c)
### Does not support numpy functions like exp or log
### The magnetic field is the same as the electic field but uses cross product of velocity in the Lorentz Force Law

def RHS(t, yvec, BF, q, m):
    pos = yvec[:3]
    vel = yvec[3:]
    Fmag = (q/m)*np.cross(vel,BF(pos))
    return vel[0], vel[1], vel[2], Fmag[0], Fmag[1], Fmag[2]

Bfield = input("Please input the B-field in the form and using variables (x,y,z): ").replace("(","").replace(")","").split(",")
yvec = input("And what is the initial position and velocity in the form (x,y,z,vx,vy,vz): ").replace("(","").replace(")","").split(",")
yvec0 = [eval(i) for i in yvec]


def BField(position):
    x = position[0]
    y = position[1]
    z = position[2]
    return np.array([eval(Bfield[0]), eval(Bfield[1]), eval(Bfield[2])])

t_end = 10

solution = solve_ivp(RHS, [0., t_end], yvec0, args=(BField, 1, 1), atol=1e-8)

BFline_x = solution.y[0, :]
BFline_y = solution.y[1, :]
BFline_z = solution.y[2, :]

ax = plt.figure().add_subplot(projection='3d')
ax.plot(BFline_x, BFline_y, BFline_z)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

plt.show()