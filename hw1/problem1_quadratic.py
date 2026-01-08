#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 15:59:11 2025

@author: evanwillmarth
"""


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12


n_elements = 2
n_nodes = 5

# Nondimensionalizing with respect to l
xcoords = np.array([[0.0, 1.0, 2.0], 
                    [2.0, 3.0, 4.0]])


# Columns represent local node numbering of each element
# The entries on each row represent the global numbering of the nodes for each element
element_nodes = np.zeros((n_elements,3),dtype=int)
for i in range(n_elements):
    for j in range(3):
        element_nodes[i,j] = i*2 + j

E = 1
A = 1
q = 1


# Calculating shape functions N
def shape_funcs(xi):
    
    N1 = 0.5*xi*(xi - 1.0)
    N2 = 1.0 - xi**2
    N3 = 0.5*xi*(xi + 1.0)
    
    return np.array([N1, N2, N3])

# Calculating derivatives of the shape functions dN/dξ
def dN_dxi(xi):
    
    dN1 = xi - 0.5
    dN2 = -2.0*xi
    dN3 = xi + 0.5
    
    return np.array([dN1, dN2, dN3])


def p_load(xi):
    # This is the distributed load defined on the local coordinate system
    p = (3/2)*q + (q/2)*xi
    return p




# Gaussian quadrature points and weights for 2-point rule
xi_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
w_pts = np.array([1.0, 1.0])

K_global = np.zeros((n_nodes,n_nodes))
F_global = np.zeros((n_nodes,1))

for i in range(n_elements):

    Ke = np.zeros((3,3))
    Fe = np.zeros((3,1))
    
    for xi, w in zip(xi_pts, w_pts):
        
        # Jacobian dx/dξ
        dNdxi = dN_dxi(xi)
        J = np.dot(dNdxi, xcoords[i,:])
        
        # dN/dx = dN/dξ * 1/J
        dNdx = dNdxi / J
        
        # B matrix = dNdx (1x3)
        B = dNdx.reshape(1,3)
        
        # Element integrand B^T * E*A * B times weight * Jacobian
        Ke += (B.T@B)*(E*A*w*np.abs(J))    
    
        Ns = shape_funcs(xi)   
        Ns = Ns.reshape(3,1)
        p = p_load(xi)
    
        if i == 0:
            Fe += Ns*p*w*np.abs(J)
    
      
    K_global[element_nodes[i,0],element_nodes[i,0]] += Ke[0,0]
    K_global[element_nodes[i,0],element_nodes[i,1]] += Ke[0,1]
    K_global[element_nodes[i,0],element_nodes[i,2]] += Ke[0,2]
    K_global[element_nodes[i,1],element_nodes[i,1]] += Ke[1,1]
    K_global[element_nodes[i,1],element_nodes[i,2]] += Ke[1,2]
    K_global[element_nodes[i,2],element_nodes[i,2]] += Ke[2,2]

    F_global[element_nodes[i,0],0] += Fe[0,0]
    F_global[element_nodes[i,1],0] += Fe[1,0]
    F_global[element_nodes[i,2],0] += Fe[2,0]
    

# Using the symmetry property of the stiffness matrix
for i in range(n_nodes-1):
    for j in range(i+1,n_nodes):
        K_global[j,i] = K_global[i,j]
        

# Applying fixed-end boundary conditions
K_global = np.delete(K_global,[0,n_nodes-1],axis=0)
K_global = np.delete(K_global,[0,n_nodes-1],axis=1)   
F_global = np.delete(F_global,[0,n_nodes-1],axis=0)

d = np.linalg.inv(K_global)@F_global

# Make a new displacement vector to include the endpoints
d_new = np.zeros(n_nodes)
d_new[0] = 0.0
d_new[1] = d[0]
d_new[2] = d[1]
d_new[3] = d[2]
d_new[4] = 0.0



# Plotting shape Functions
xi = np.linspace(-1.0, 1.0, 200)
d_phys_total = []
x_phys_total = []
plt.figure()
for ii in range(n_elements):
    # Element  physical coordinates
    x_elem = xcoords[ii]
    d_elem = np.array([d_new[2*ii],d_new[2*ii+1],d_new[2*ii+2]])
    x_phys = []
    d_phys = []
    Nvals = []
    for x in xi:
        Nxi = shape_funcs(x)
        x_phys.append(np.dot(Nxi, x_elem))
        d_phys.append(np.dot(Nxi, d_elem))
        Nvals.append(Nxi)
    
    x_phys = np.array(x_phys)
    Nvals = np.array(Nvals)
    d_phys_total.append(d_phys)
    x_phys_total.append(x_phys)
    
    
    if ii == 0:
        plt.plot(x_phys, Nvals[:,0],color='green', label=r'$N_1(x)^%d$'%(ii+1))
        plt.plot(x_phys, Nvals[:,1],color='green',linestyle='dashed', label=r'$N_2(x)^%d$'%(ii+1))
        plt.plot(x_phys, Nvals[:,2],color='green',linestyle='dashdot', label=r'$N_3(x)^%d$'%(ii+1))
    else:
        plt.plot(x_phys, Nvals[:,0],color='red', label=r'$N_1(x)^%d$'%(ii+1))
        plt.plot(x_phys, Nvals[:,1],color='red',linestyle='dashed', label=r'$N_2(x)^%d$'%(ii+1))
        plt.plot(x_phys, Nvals[:,2],color='red',linestyle='dashdot', label=r'$N_3(x)^%d$'%(ii+1))
        
        
plt.xlabel('x')
plt.ylabel('Shape function value')
plt.title('Shape Functions in Physical Coordinates (Both Elements)')
plt.legend()
plt.grid(True)
plt.show()
    

# Plotting Displacements
plt.figure()
for ii in range(n_elements):
    if ii == 0:
        plt.plot(x_phys_total[ii], d_phys_total[ii], color='green')
    if ii == 1:
        plt.plot(x_phys_total[ii], d_phys_total[ii], color='red')

plt.xlabel('x')
plt.ylabel(r'$\frac{l^2q}{C}$')
plt.title('Displacements in Global Coordinates (All Elements)')








