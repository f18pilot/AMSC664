#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 3 16:31:00 2023

@author: bscheufele

###################################################################################
#   Project: Using Periodic Structure in Waveguide to amplify RF signal           #
#   Program: Solving the quartic for a smooth waveguide. This program can be run  #
#            standalone, where it computes the quartic and full sim for the       #
#            waveguide and plots the comparison for a range of magnetic field     #
#            values sequentially, or it can be used as a quartic solver for a     #
#            a particular frequency, given the parameters required.               #
#   Author:  B Scheufele                                                          #
#                                                                                 #
#   Version: 8.0 (Apr 2023)                                                       #
#                                                                                 #
#   Dependencies: Scipy, numpy, Apr23_ode.py, Apr23_plots.py                      #
#                                                                                 #
#   Approach: The write up for this program is similar to "AMSC663 Research       #
#             Midterm Report Dec 2022", where it solves the system equations      #
#             for a set of initial conditions, including I_beam = 0.4 ~ 500A      #
#             The power is calculated and plotted, and the slope of the           #
#             power curve is calculated and plotted.  This is to be compared      #
#             to the solution to the quartic equation which solves for beta.      #
#             *These roots are plotted and compared to the simulated solution.    #
#                                                                                 #
###################################################################################
"""
# This file computes the polynomial from inputted values for the dispersion relation

from scipy import constants as cn
import numpy as np
import fqs
import matplotlib.pyplot as plt


def quartic_solver(k_0, gamma_p, rho_z_ic, rho_perp_ic, K_0, I_hat, W, gamma):
    """ This function solves the equation:
                    A x^4 + Bx^3 + C x^2 + D x + E = 0
    :param k_0: wavenumber of the frequency
    :param gamma_p: relativistic factor of the electrons
    :param rho_z_ic: initial normalized momentum in the z direction
    :param rho_perp_ic: initial normalized momentum in the perpendicular direction
    :param K_0: interaction term, set to a constant (average) value for range
    :param I_hat: normalized beam current
    :param W: cyclotron term
    :param gamma: factor for how far frequency is from cutoff
    :return:  root_list_final_imag, root_list_final_real ; imaginary and real parts of beta for inputted frequency
    """

    # Form the coefficients of the quartic
    A_q = 1
    B_q = -2 * W
    C_q = W ** 2 - k_0 ** 2 * gamma + I_hat * K_0 ** 2 * rho_perp_ic ** 2 / rho_z_ic ** 3 + 2 * I_hat * \
          K_0 ** 2 / rho_z_ic
    D_q = 2 * W * k_0 ** 2 * gamma - 2 * gamma_p * k_0 * I_hat * K_0 ** 2 / rho_z_ic ** 2 - 2 * W * \
          I_hat * K_0 ** 2 / rho_z_ic
    E_q = - W ** 2 * k_0 ** 2 * gamma - k_0 ** 2 * I_hat * K_0 ** 2 * (
            rho_perp_ic ** 2 / rho_z_ic ** 3) + 2 * k_0 * I_hat * K_0 ** 2 * W * gamma_p / rho_z_ic ** 2

    # Solve the quartic using fqs (can also use python quartic solvers)
    roots = fqs.quartic_roots([A_q, B_q, C_q, D_q, E_q])

    # Test the roots to find the complex conjugates (should be 2 real roots and 2 complex conjugates, or 4 real)

    a = 4
    for jj in range(4):
        test_root = roots[0][jj]
        if np.imag(test_root) != 0:
            a -= 1
            if np.imag(test_root) < 0:
                a -= 1
                if abs(np.imag(test_root)):
                    root_list_final_imag = abs(np.imag(test_root))
                    root_list_final_real = np.real(test_root)

    if a == 4:
        # For this frequency there are all real roots, this is a gap in the dispersion curve
        root_list_final_imag = np.imag(0)
        root_list_final_real = np.real(0)

    return root_list_final_imag, root_list_final_real


if __name__ == '__main__':
    # This program, if run as a stand alone, will calculate the quartic and the full system simulation for
    # comparison at several different magnetic field strengths.  This allows the user to fine tune the cyclotron
    # dispersion to maximize the wave amplitude gain. First, define the input variables
    N = 5
    gamma_p = np.sqrt(1 + 2 * 1.22 ** 2)
    rho_z_ic = 1.22
    rho_perp_ic = 1.22
    w = 0.05
    K_0 = 14.14  # 1/sqrt(w*h), h = 0.05
    I_hat = 0.4  # normalized beam current (multiply by 1356 to get current in amps)

    # If you are comparing to a corrugated scenario, choose the scenario's cutoff frequency, omega_c
    omega_c = 18886924854.0

    # This following section is to help determine the best magnetic field strength for a given x_c
    Omega_B_ = np.linspace(2.75 * 10 ** 10, 5.0 * 10 ** 10, 10)
    for jj in range(len(Omega_B_)):
        # Set up the run
        Omega_B = Omega_B_[jj]
        omega_points = np.linspace(1.9 * 10 ** 10, 6.5 * 10 ** 10, 100)       # Desired range of frequencies
        roots_list = []
        root_list_final_imag = []
        root_list_final_real = []
        for ii in range(len(omega_points)):
            # Calculate needed variables for the quartic equation
            omega = omega_points[ii]
            k_0 = omega / cn.c
            W = (omega * gamma_p - Omega_B) / (cn.c * rho_z_ic)
            gamma = 1 - omega_c ** 2 / omega ** 2

            # Solve the quartic equation for each frequency
            root_imag, root_real = quartic_solver(k_0, gamma_p, rho_z_ic, rho_perp_ic, K_0, I_hat, W, gamma)

            # Store the roots in a list for plotting
            root_list_final_real.append(root_real)
            root_list_final_imag.append(root_imag)

        # Plot the comparison for each value of the magnetic field (cyclotron frequency) to find best value
        fig, (ax2) = plt.subplots(1)
        fig.suptitle(f'Roots (Real and Imaginary) vs omega for Omega_B = {Omega_B:.2e}')

        # Calculate and plot cyclotron equation
        beta_real = np.linspace(0, 120, 200)
        omega_lin = [x * cn.c * rho_z_ic / gamma_p + Omega_B / gamma_p for x in beta_real]
        ax2.plot(beta_real, omega_lin, 'g.', label='cyclotron')

        # Calculate and plot smooth waveguide dispersion relation and cutoff frequency
        omega_dispersion = [np.sqrt(omega_c ** 2 + cn.c ** 2 * x ** 2) for x in beta_real]
        ax2.plot(beta_real, omega_dispersion, 'b.', label='Dispersion relation')
        ax2.plot(0, omega_c, 'ro', label='cutoff frequency')

        # Plot the roots as a function of wavenumber
        root_list_final_imag_neg = [x for x in root_list_final_imag]
        ax2.plot(root_list_final_imag, omega_points, 'r+', label=r'$ \beta_{imag}$ from quartic')
        ax2.plot(root_list_final_real, omega_points, 'm+', label=r'$ \beta_{real}$ from quartic')
        ax2.set_ylabel('Omega [radians/sec]')
        ax2.set_xlabel(r'$ \beta_{real}$ / $ \beta_{imag}$ [$cm^{-1}]$')
        ax2.set_ylim(0, 4.5*10**10)
        ax2.set_xlim(-5, 120)
        plt.legend()
        plt.show()
