#!/usr/bin/env python
# coding: utf-8

#Importing relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

#Including numba
#%pip install numba
import numba as nb
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

#Defining relevant parameters
tilting_angle = np.radians(23.4)      #Axial tilt of Earth, i.e. angle between rotational axis and orbital axis, [tilting_angle] = rad.
B0 = 1                                #Mean magnitude of the Earth's magnetic field at the equator, [B0] = T

'''
(1) Calculate the magnetic field
----------------------------------------------------------------------------------------------------------------------------------
'''

#Setting up meshgrid
X = np.linspace(-10,10,500)
Y = np.linspace(-10,10,500)
Z = np.linspace(-10,10,500)
xx, yy = np.meshgrid(X,Y)
xx, zz = np.meshgrid(X,Z)

@jit(nopython=True)
def f(theta,phi):
    '''
    Parameters:
        theta, phi = Angles in spherical coordinates
    Output:
        Directional vector
        
    Function used to find the direction of the magnetic moment.
    '''

    xx = np.sin(theta)*np.cos(phi) 
    yy = np.sin(theta)*np.sin(phi) 
    zz = np.cos(theta) 

    return np.array([xx,yy,zz])

#Direction of the magnetic moment
m_hat = f(tilting_angle,0)

#@jit(nopython=True)
def B(X,Y,Z):
    '''
    Parameters:
        X,Y,Z = Meshgrid
    Output:
        The magnetic field for the given meshgrid.
        
    Function that uses calculates the magnetic field using the dipole model.
    '''

    R = np.sqrt(X**2 + Y**2 + Z**2)
    r_vec = np.array([X,Y,Z], dtype = np.dtype('f4'))
    
    B1 = np.sum(r_vec * m_hat[:,None,None],axis = 0)/R**5
    B2 = m_hat[:,None,None]/R**3

    return B0 * (3 * B1 * r_vec - B2)

#Caclulating the magnetic field
Bx, By,Bz = B(xx,yy,zz)

#Plotting the xz and xy plane
fig1, ax1 = plt.subplots(1,2,figsize = (15,7))

ax1[0].set_title('Magnetic field around Earth', fontsize = 20)
ax1[0].streamplot(xx, zz, Bx,Bz)
ax1[0].plot(np.zeros(len(Y)),Y, 'k--')
ax1[0].plot(tilting_angle*X,Y, 'r', label = 'Axis of rotation')
ax1[0].plot(X,np.zeros(len(X)), 'k--')
ax1[0].plot(-(tilting_angle + np.pi/2)*X,Y, 'g', label = 'Equator')
ax1[0].legend()
ax1[0].add_patch(Circle((0,0), 1, color='b', zorder=100))
ax1[0].set_xlabel('$x$', fontsize = 15)
ax1[0].set_ylabel('$z$', fontsize = 15)
ax1[0].set_xlim(-10, 10)
ax1[0].set_ylim(-10, 10)
ax1[0].set_aspect('equal')

ax1[1].set_title('Magnetic field around Earth', fontsize = 20)
ax1[1].streamplot(xx, yy, Bx,By)
ax1[1].add_patch(Circle((0,0), 1, color='b', zorder=100))
ax1[1].plot(np.zeros(len(Y)),Y, 'k--')
ax1[1].plot(X,np.zeros(len(X)), 'k--')
ax1[1].set_xlabel('$x$', fontsize = 15)
ax1[1].set_ylabel('$y$', fontsize = 15)
ax1[1].set_xlim(-10, 10)
ax1[1].set_ylim(-10, 10)
ax1[1].set_aspect('equal')

plt.show()

'''
(2) Model solar wind
----------------------------------------------------------------------------------------------------------------------------------
'''
#Defining constants
q = 1
m = 1
v0 = 1

@jit(nopython=True)
def B_not_mesh(pos):
    '''
    Parameters:
        pos = 3D vector containing the position at one point in the grid.
    Output:
        Magnetic field around a single point.
        
    Function that finds the magnetid field relative to a particle moving in it. Uses the same formula as in the
    function B, but the output is not a meshgrid but a 3D vector.
    '''
    
    X,Y,Z = pos
    
    R = np.sqrt(X**2 + Y**2 + Z**2)
    r_vec = np.array([X,Y,Z])
    
    B1 = (r_vec[0]*m_hat[0] + r_vec[1]*m_hat[1] + r_vec[2]*m_hat[2])/R**5
    B2 = m_hat/R**3

    return B0 * (3 * B1 * r_vec - B2)

@jit(nopython=True)
def f(motion):
    ''' 
    Parameters:
        motion = Position and velocity of a particle at a step in the ode_solver
    Output:
        The RHS to the differential equations describing the equations of motion.
        
    Function that solves the differential equation for one step.
    '''

    B_at_pos = B_not_mesh(motion[0:3])
    
    dt_dmotion = np.zeros(6)
    dt_dmotion[0:3] = motion[3:6]

    dt_dmotion[3:6] = q/m * np.cross(motion[3:6],B_at_pos)
    
    return dt_dmotion

@jit(nopython=True)
def RK4(f, motion, dt): 
    '''
    Parameters:
        f = Differential equations
        motion = Previous step.
        dt = Timestep
    Output:
        motion_next = Next step.
        
    One step in time using Runge Kutta of the 4th order.
    '''
    k1 = f(motion)
    k2 = f(motion + dt*(k1/2))
    k3 = f(motion + dt*(k2/2))
    k4 = f(motion + dt*k3)

    motion_next = motion + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    return motion_next

@jit(nopython=True)
def ode_solver(f, method, motion_init, t_end, N):
    '''
    Parameters:
        f = Differential equations
        method = Method used to solve differential equations
        motion_init = Initial positions and velocity
        t_end = Runtime
        N = Number of steps
    Output:
        timearray
        full_motion = Solution to differential equations as solved by the method
    
    Function that solves the differential equation using a given method.
    '''

    timearray = np.linspace(0, t_end, N)
    dt = t_end/N
    full_motion = np.zeros((len(timearray),6))
    full_motion[0] = motion_init

    
    for i in range(1, len(full_motion)):
        full_motion[i] = method(f, full_motion[i-1], dt)

          
    return timearray, full_motion

#Initial position and velocity for the particle
motion_init1 = np.array([-10,0,0,v0,0,0])   #Initial position far from Earth
motion_init2 = np.array([-0.1,0,0,v0,0,0])  #Initial position close to Earth

#Can comment these lines out after running and saving to file
timearray1, full_motion1 = ode_solver(f,RK4, motion_init1, 40, 1000000)
np.savetxt('motion1.txt', full_motion1[:,0:3])
timearray2, full_motion2 = ode_solver(f,RK4, motion_init2, 40, 1000000)
np.savetxt('motion2.txt', full_motion2[:,0:3])

#Reading from file
motion1_file = np.loadtxt('motion1.txt')
motion2_file = np.loadtxt('motion2.txt')

fig2 = plt.figure(figsize = (15,7))
ax21 = fig2.add_subplot(1, 2, 1, projection='3d')
ax22 = fig2.add_subplot(1, 2, 2, projection='3d')

ax21.plot(motion1_file[:,0], motion1_file[:,1], motion1_file[:,2])
ax21.set_xlabel('X')
ax21.set_ylabel('Y')
ax21.set_zlabel('Z')

ax22.plot(motion2_file[:,0], motion2_file[:,1], motion2_file[:,2])
ax22.set_xlabel('X')
ax22.set_ylabel('Y')
ax22.set_zlabel('Z')

plt.show()

'''
(3) Accuracy for numerical model
----------------------------------------------------------------------------------------------------------------------------------
'''

dt_num = np.logspace(-7, -1, 50)

motion_init = np.array([-0.1,0,0,v0,0,0])
Ke_initial = 0.5*m*(motion_init[3]*motion_init[3] + motion_init[5]*motion_init[5] +motion_init[5]*motion_init[5])

@jit(nopython=True)
def numerical_validity(dt, Ke_init, motion_init):
    '''
    Parameters:
        dt = Array of timesteps
        Ke_init, motion_init = Inital position, velocity and kinetic energt
    Output:
        deviations = Array of deviation between calculated and initial kinetic energy
        
    Function that compares the calculated kinetic energy to the initial one in order to check the numerical
    validity for an array of timesteps.
    '''

    deviations = np.zeros(len(dt_num))

    for i in range(len(dt_num)):
        t_end = 1
        num_steps = int(t_end/dt_num[i])

        timeArray, full_motion = ode_solver(f,RK4, motion_init, t_end, num_steps)

        Ke = 0.5*m*(full_motion[num_steps-1,3]*full_motion[num_steps-1,3] + full_motion[num_steps-1,4]*full_motion[num_steps-1,4] + full_motion[num_steps-1,5]*full_motion[num_steps-1,5])

        deviations[i] = np.abs(Ke - Ke_initial)
        
    return deviations

#Can comment these lines out after running and saving to file
dev = numerical_validity(dt_num, Ke_initial, motion_init)
np.savetxt('num_dev.txt', dev)

#Reading from file
dev_file = np.loadtxt('num_dev.txt')

fig2 = plt.figure(figsize = (13,7))
plt.plot(dt_num, dev_file)
plt.grid()
plt.title('Numerical validity', fontsize = 25)
plt.xlabel('dt', fontsize = 20)
plt.ylabel('Ke - Ke_init', fontsize = 20)
plt.show()

fig3 = plt.figure(figsize = (13,7))
plt.loglog(dt_num, dev_file)
plt.grid()
plt.title('Numerical validity - Logarithmic scale', fontsize = 25)
plt.xlabel('dt', fontsize = 20)
plt.ylabel('Ke - Ke_init', fontsize = 20)
plt.show()