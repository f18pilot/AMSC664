#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 3 09:31:00 2022

@author: bscheufele

###################################################################################
#   Project: Using Periodic Structure in Waveguide to amplify RF signal           #
#   Program: ODE to solve waveguide system for no particles periodic waveguide    #
#   Author:  B Scheufele                                                          #
#                                                                                 #
#   Version: 3.0 (2 Apr 2023 updated)                                             #
#                                                                                 #
#   Dependencies: scipy, numpy, matplotlib                                        #
#                                                                                 #
#   Approach: The approach is described in "Formulating Wave-Beam                 #
#   interaction equations",  24 March 2022, which builds on the method of the     #
#   M. Botton, et al., “MAGY: A Time-Dependent Code for Simulation of Slow        #
#   and Fast Microwave Sources” , 1998 paper. These equations describe one        #
#   mode (TE_10), and calculate the interaction between the mode and all          #
#   the particles.   This ode solver is for the corrugated waveguide.             #
#                                                                                 #
###################################################################################

"""
from scipy import constants as cn
import numpy as np

global ST0_list
ST0_list = []


def motion(r_0, z, omega, initial_conditions):
    """
    This function includes the governing equations for both wave/particle in the smooth waveguide for use in the ODE

    :param r_0: state vector of the system [I_0, V_0, rho_perp(i), rho_z(i), psi(i) ], for each particle (i)
    :param z: vector of z points, z is the axial length, np.linspace(0, L, step)
    :param omega: frequency in radians/sec
    :param initial_conditions = [N, n_freq, steps, g, h_0, w, L, omega_c, I_b, Omega_B, B_0, k_b]

    :return: r_0, state vector of the system, see above
    """

    # Set the state vector and frequency
    r = r_0
    # print('r_0', r, r_0)
    k = omega / cn.c

    # Set the initial conditions
    N = initial_conditions[0]                   # number of particles in the simulation
    n_freq = initial_conditions[1]              # number of frequencies selected to simulate
    steps = initial_conditions[2]               # number of z points, steps for integration
    g = initial_conditions[3]                   # g = gamma/2; gamma = 0  is the period of smooth waveguide
    h_0 = initial_conditions[4]                 # height of the waveguide (constant for smooth waveguide)
    w = initial_conditions[5]                   # width of the waveguide
    L = initial_conditions[6]                   # length of the waveguide
    omega_c = initial_conditions[7]             # cutoff frequency of the waveguide
    I_b = initial_conditions[8]                 # current of the particle beam, in dimensionless units
    rho_z_ic = initial_conditions[9]         # initial value of rho_perp
    rho_perp_ic = initial_conditions[10]           # initial value of rho_z
    Omega_b = initial_conditions[11]            # cyclotron frequency
    k_b = initial_conditions[12]                # wavenumber of the particle beam
    ell = initial_conditions[13]
    step = initial_conditions[14]               # the frequency index

    # Set the initial conditions
    I_0_hat = r[0]  # wave current, dimensionless units
    V_0_hat = r[1]  # wave voltage, dimensionless units
    I_beam = I_b  # particle beam current, dimensionless units
    K_0_ = K_0(z, initial_conditions)  #
    gamma = np.sqrt(1 - omega_c ** 2 / omega ** 2)

    # intermediate lists - for every particle
    rho_perp = []
    rho_z = []
    psi = []

    for j in range(2, 3 * N, 3):
        rho_perp.append(r[j])
        rho_z.append(r[j + 1])
        psi.append(r[j + 2])

    # final list
    r_ = []

    ST0 = S_T0(I_beam, z, rho_perp, rho_z, psi, initial_conditions)
    # To analyze the ST0 parameter, keep track of it during algorithm as a global variable
    ST0_list.append(ST0)

    # Calculate V and I, (r1 = I, r2 = V)
    r1 = 1j * k * gamma ** 2 * V_0_hat - K_00(z, initial_conditions) * I_0_hat - ST0
    r2 = 1j * k * I_0_hat + K_00(z, initial_conditions) * V_0_hat

    r_.append(r1)
    r_.append(r2)

    # Begin the particle calculations
    if rho_z or rho_perp:
        if rho_z or rho_perp > 10 ** (-10):
            for i in range(N):
                # this solves for all particles: p_perp, p_z, and psi
                gamma_p = np.sqrt(1 + (rho_perp[i] ** 2 + rho_z[i] ** 2))

                r3 = (gamma_p / rho_z[i]) * np.real(np.exp(1j * psi[i] - 1j * k_b * z) * K_0_ *
                                                    (V_0_hat - (rho_z[i] / gamma_p) * I_0_hat))
                r4 = (rho_perp[i] / rho_z[i]) * np.real(np.exp(1j * psi[i] - 1j * k_b * z) * K_0_ * I_0_hat)
                r5 = (k_b - omega * gamma_p / (cn.c * rho_z[i]) + Omega_b / (rho_z[i] * cn.c) - np.real(
                    np.exp(1j * psi[i] - 1j * k_b * z) * (-1j * K_0_) *
                    ((gamma_p * V_0_hat) / (rho_perp[i] * rho_z[i]) - (1 / rho_perp[i]) * I_0_hat)))

                r_.append(r3)
                r_.append(r4)
                r_.append(r5)
        else:
            r_.append(0)
            r_.append(0)
            r_.append(0)

    return r_


def K_00(z, initial_conditions):
    """
    This function os determined solely by the structure of the waveguide
            h(z) = h_0 * some function for the structure, h_0 is constant for smooth waveguide
            K_00 = -1/2h dh(z)/dz
            K_00 = +/- gamma_h/2, gamma_h = 2* g/ell, where ellis the period of the structure
    For smooth structure, K_00 = 0
    :param z: axial coordinate
    :param g: related to period of structure
    :param L: length of structure
    :return: K_00
    """
    # # Set the initial conditions
    # # initial_conditions = [N, n_freq, steps, g, h_0, w, L, omega_c, I_b, rho_perp_ic, rho_z_ic, Omega_B, B_0, k_b, ell, jj]
    # N = initial_conditions[0]  # number of particles in the simulation
    # n_freq = initial_conditions[1]  # number of frequencies selected to simulate
    # steps = initial_conditions[2]  # number of z points, steps for integration
    # g = initial_conditions[3]  # g = gamma/2; gamma = 0  is the period of smooth waveguide
    # h_0 = initial_conditions[4]  # height of the waveguide (constant for smooth waveguide)
    # w = initial_conditions[5]  # width of the waveguide
    # L = initial_conditions[6]  # length of the waveguide
    # omega_c = initial_conditions[7]  # cutoff frequency of the waveguide
    # I_b = initial_conditions[8]  # current of the particle beam, in dimensionless units
    # Omega_b = initial_conditions[9]  # cyclotron frequency
    # B_0 = initial_conditions[10]  # magnetic field strength
    # k_b = initial_conditions[11]  # wavenumber of the particle beam
    # ell = initial_conditions[14] period of waveguide
    # step = initial_conditions[15]  # the frequency index

    L = initial_conditions[6]
    g = initial_conditions[3]
    ell = initial_conditions[14]
    if ell == 0:
       K_00 = 0
    else:
        gamma_h = 2*g/ell
        if 0 <= np.mod(z, L) < L / 2:
            K_00 = -gamma_h / 2
        elif L / 2 <= np.mod(z, L) <= L:
            K_00 = gamma_h / 2

    return K_00


def K_0(z, initial_conditions):
    """
    This function is a coupling parameter in the beam equation.
    It places the beam along the axis when cos(pi*Y/w)=1
            K_0 = sqrt(2/(w*h(z)) cos(pi/w *Y)
            K_0' = -1j*K_0

    :param z: axial coordinate
    :param initial_conditions: g, h_0, w, L
    :return: K_0
    """
    # Set the initial conditions
    g = initial_conditions[3]
    h_0 = initial_conditions[4]
    w = initial_conditions[5]
    L = initial_conditions[6]

    gamma_h = 2*g/L
    if 0 <= np.mod(z, L) < L / 2:
        h = h_0 * np.exp(gamma_h * np.mod(z, L))

    elif L / 2 <= np.mod(z, L) <= L:
        h = h_0 * np.exp(-gamma_h * (np.mod(z, L) - L))

    K_0 = np.sqrt(1/(2*w*h))

    return K_0


def S_T0(I_beam_hat, z, rho_perp, rho_z, psi, initial_conditions):
    """
    ST0 is the coupling term between the wave and particle beam. It is dimensionless.
        ST0 = 2/N * I_hat_beam * exp(ik_bz) * SUM (over N) rho_perp(i)/rho_z(i) *exp( -1j*psi(i))*conj(K_0)
    The conjugate of K_0 is the same as K_0 for the smooth waveguide.

    :param I_beam_hat: beam current, dimensionless
    :param z: axial coordinate
    :param rho_perp: dimensionless momentum of particle in direction perpendicular to axial direction
    :param rho_z: dimensionless momentum of particle in the z direction
    :param psi: phase of the particle
    :param initial_conditions: N, g, h_0, w, L, k_b (see below)
    :return: ST0
    """

    # Set the initial conditions
    N = initial_conditions[0]    # number of particles in the simulation
    g = initial_conditions[3]    # g = gamma/2; gamma = 0  is the period of smooth waveguide
    h_0 = initial_conditions[4]  # height of the waveguide (constant for smooth waveguide)
    w = initial_conditions[5]    # width of the waveguide
    L = initial_conditions[6]    # length of the waveguide
    k_b = initial_conditions[13] # wavenumber of the particle beam

    if N == 0:
        # print('No particles!')
        return 0
    else:
        sum_p = 0 + 0 * 1j
        # Calculate the sum over all particles
        for i in range(N):
            rho_P = rho_perp[i]
            rho_Z = rho_z[i]
            Psi = psi[i]
            sum_p = sum_p + (rho_P / rho_Z) * np.exp(-1j * Psi) * K_0(z, initial_conditions)

        return 2 * I_beam_hat * np.exp(1j * k_b * z) * (1 / N) * sum_p


# rk4 ode algorithm
def ode_motion(r_, z_points, omega, particle_dict, initial_conditions):
    """
    This is the RK4 method of solving a system of linear equations.  The method is written as outlined in
    "Computational Astrophysics", by Mark Neuman, 2013

    :param r_: state vector for the system
    :param z_points: set of points along the axial (z) axis
    :param omega: frequency in radians/sec
    :param initial_conditions: N, n_freq, steps, g, h_0, w, L, omega_c, I_b, Omega_B, B_0, k_b, jj, see below
    :return: r, particle_dict, ST0_list(global variable)
    """
    # Set the initial conditions
    N = initial_conditions[0]           # number of particles in the simulation
    steps = initial_conditions[2]       # number of z points, steps for integration
    L = initial_conditions[6]           # length of the waveguide
    step = initial_conditions[14]       # the frequency index

    # Prepare the data and determine step size
    h = L/(steps)

    r = [item for element in r_ for item in element]  # converts list of lists to list of floats
    #print('r inside ode', r)

    # Begin the ode algorithm
    for j in range(len(z_points)):
        z = z_points[j]

        k1 = h * np.array(motion(r, z, omega, initial_conditions), dtype=object)
        k2 = h * np.array(motion(r + 0.5 * k1, z + 0.5 * h, omega, initial_conditions),
                          dtype=object)
        k3 = h * np.array(motion(r + 0.5 * k2, z + 0.5 * h, omega, initial_conditions),
                          dtype=object)
        k4 = h * np.array(motion(r + k3, z + h, omega, initial_conditions), dtype=object)

        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6


        # Add to the particle dictionary

        particle_dict[f"I_0_{step}"].append(r[0])
        particle_dict[f"V_0_{step}"].append(r[1])

        if N != 0:
            for i in range(N):
                a = 2 + i * 3
                b = 3 + i * 3
                c = 4 + i * 3
                particle_dict[f"p_perp{step}{i}"].append(r[a])
                particle_dict[f"p_z{step}{i}"].append(r[b])
                particle_dict[f"psi{step}{i}"].append(r[c])
    #print('r after ode', r)
    return r, particle_dict, ST0_list
