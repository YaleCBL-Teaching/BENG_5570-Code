#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 10:49:26 2026

@author: evanwillmarth
"""





import numba
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import scipy

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12




# Define rows and width of the lattice
rows = 24
width = 24
n_nodes = rows*width
# r is the nx2 array containing the global nodal coordinates.
r = np.zeros((n_nodes,2))
# a is the lattice spacing
a = 1.0

# Periodic Box size dimensionss
ly = 12*np.sqrt(3)
lx = 24

#
k = 0
for i in range(rows):
    for j in range(width):
        if i != 0 and i%2 != 0:
            r[k,0] = a*j + a/2
            r[k,1] = i*(np.sqrt(3)/2)*a
            k += 1
        else:
            r[k,0] = a*j 
            r[k,1] = i*(np.sqrt(3)/2)*a
            k += 1

# Center the periodic box around the origin.
r[:,0] = r[:,0]/lx
r[:,1] = r[:,1]/ly
r[:,0] = r[:,0] - np.rint(r[:,0])
r[:,1] = r[:,1] - np.rint(r[:,1])
r[:,0] = r[:,0]*lx
r[:,1] = r[:,1]*ly



# Determine the nodal connectivty of the network by
# checking wich nodes are within a lattice spacing apart from eachother.
element_nodes = np.zeros((3*n_nodes,2),dtype=int)
rij = np.zeros(2)
k = 0
for i in range(n_nodes-1):
    for j in range(i+1, n_nodes):
        rij[0] = r[i,0] - r[j,0]
        rij[1] = r[i,1] - r[j,1]
        rij[0] = rij[0]/lx
        rij[1] = rij[1]/ly
        rij[0] = rij[0] - np.rint(rij[0])
        rij[1] = rij[1] - np.rint(rij[1])
        rij[0] = rij[0]*lx
        rij[1] = rij[1]*ly 
        
        if np.abs(np.linalg.norm(rij)) < 1.1:
            element_nodes[k,0] = i
            element_nodes[k,1] = j
            k += 1




    

##################################################################################
##################################################################################

A = 1
E = 1
strain = 0.0
n_elements = len(element_nodes[:,0])

K_global = np.zeros((2*n_nodes,2*n_nodes))
element_lengths = np.zeros(n_elements)

elem_angles = np.zeros(n_elements)
for ii in range(n_elements):
    
    i = element_nodes[ii,0]
    j = element_nodes[ii,1]

    dx = r[j, 0] - r[i, 0]
    dy = r[j, 1] - r[i, 1]
    
    # Enforcing Periodic Boundary conditions.
    # Number of y-box separations
    ny = np.rint(dy / ly)
    # Minimum-image y
    dy_min = dy - ny * ly
    # Shear offset
    s = strain * ly  # where strain = γ(t)
    # Sheared x-separation before wrapping
    dx_shear = dx - ny * s
    # Standard x wrap
    dx_min = dx_shear - np.rint(dx_shear / lx) * lx
    
    x_dif = dx_min
    y_dif = dy_min

    elem_angles[ii] = np.arctan2(y_dif,x_dif)
    L = np.sqrt(x_dif**2 + y_dif**2)
    element_lengths[ii] = L

    cs_array = np.array([[-np.cos(elem_angles[ii]),-np.sin(elem_angles[ii]),
                          np.cos(elem_angles[ii]),np.sin(elem_angles[ii])]])
    
    Ke = (A*E/L)*np.outer(cs_array, cs_array)
    
    K_global[2*element_nodes[ii,0],2*element_nodes[ii,0]] += Ke[0,0]
    K_global[2*element_nodes[ii,0],2*element_nodes[ii,0]+1] += Ke[0,1]
    K_global[2*element_nodes[ii,0]+1,2*element_nodes[ii,0]+1] += Ke[1,1]
    K_global[2*element_nodes[ii,0],2*element_nodes[ii,1]] += Ke[0,2]
    K_global[2*element_nodes[ii,0],2*element_nodes[ii,1]+1] += Ke[0,3]
    K_global[2*element_nodes[ii,0]+1,2*element_nodes[ii,1]] += Ke[1,2]
    K_global[2*element_nodes[ii,0]+1,2*element_nodes[ii,1]+1] += Ke[1,3]
    K_global[2*element_nodes[ii,1],2*element_nodes[ii,1]] += Ke[2,2]
    K_global[2*element_nodes[ii,1],2*element_nodes[ii,1]+1] += Ke[2,3]
    K_global[2*element_nodes[ii,1]+1,2*element_nodes[ii,1]+1] += Ke[3,3]
    
   
# From Symmetry of the Global Stiffness Matrix
for i in range(2*n_nodes-1):
    for j in range(i+1,2*n_nodes):
        K_global[j,i] = K_global[i,j]
        
        
        
# Force Dipole
f_vec = np.zeros(2*n_nodes)
f_vec[0] = -0.001
f_vec[2] =  0.001

        
# Invert the global stiffness matrix 
u_vec = np.linalg.pinv(K_global)@f_vec

u_plot = np.zeros((n_nodes,2))
k = 0
for i in range(n_nodes):
    for j in range(2):
        u_plot[i,j] = u_vec[k]
        k += 1

# Subtract off the average motion
u_plot[:,0]  = u_plot[:,0] - np.mean(u_plot[:,0])
u_plot[:,1]  = u_plot[:,1] - np.mean(u_plot[:,1])


# Plotting of the displacements
fig, ax = plt.subplots()
fig.set_size_inches(6.0,6.0)
ax.set_aspect('equal')
quivers = []
vec = ax.quiver(r[:,0] , r[:,1], u_plot[:,0], u_plot[:,1], width=.005, scale = .001)
quivers.append(vec)


        
    
        
#####################################################################################
#####################################################################################
#####################################################################################
# Plot the bonds of the lattice


fig, ax = plt.subplots()
jjj = 0
for ii in range(len(element_nodes[:,0])):

        i = element_nodes[ii,0]
        j = element_nodes[ii,1]
        
        
        dx = r[j, 0] - r[i, 0]
        dy = r[j, 1] - r[i, 1]
        
        # Implement Periodic Boundary Conditions
        # Number of y-box separations
        ny = np.rint(dy / ly)
        # Minimum-image y
        dy_min = dy - ny * ly
        # Shear offset
        s = strain * ly  # where strain = γ(t)
        # Sheared x-separation before wrapping
        dx_shear = dx - ny * s
        # Standard x wrap
        dx_min = dx_shear - np.rint(dx_shear / lx) * lx
        xdif = dx_min
        ydif = dy_min
        

        plt.plot(np.array([r[i,0], r[i,0]+xdif]),np.array([r[i,1], r[i,1]+ydif]),'k', linewidth=1.0, alpha=0.5 )
        jjj += 1

border_x = np.linspace(-lx/2,lx/2,20) 
border_y = np.linspace(-ly/2,ly/2,20)
ax.plot(-lx/2*np.ones(20), border_y, 'k', linewidth=3)
ax.plot(lx/2*np.ones(20), border_y, 'k', linewidth=3)
ax.plot(border_x, -ly/2*np.ones(20), 'k', linewidth=3)
ax.plot(border_x, ly/2*np.ones(20), 'k', linewidth=3)













