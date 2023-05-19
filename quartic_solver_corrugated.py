#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 5 16:31:00 2022

@author: bscheufele

###################################################################################
#   Project: Using Periodic Structure in Waveguide to amplify RF signal           #
#   Program: Solving quartic equation with beta_corr values. This script can be   #
#            called as a function to solve the corrugated quartic, or it can be   #
#            run as a stand alone program to compare a particular solution to     #
#            the equation for beta_corr given in find_beta_corr.py.  From the     #
#            output of that program, you get the g amd x_c values and the         #
#            the integer-key value of the dictionary where the values for the     #
#            frequency, phi and beta_corr are stored. You must manually put in    #
#            the values of g, x_c and integer_key.                                #
#            The output is a comparison plot of the quartic and simulated         #
#            solutions.                                                           #
#   Author:  B Scheufele                                                          #
#                                                                                 #
#   Version: 5.0 (Updated 19 Apr 2023)                                                                 #
#                                                                                 #
#   Dependencies: Scipy, numpy, matplotlib, fqs                                   #
#                                                                                 #
#   Approach: Following the approach in overleaf "Quartic solution for            #
#             Corrugated Waveguide Part III", values of beta_corr from            #
#             'find_beta_corr.py' are fed into this quartic solver and            #
#             the dispersion relation is plotted and compared to the full         #
#             simulation.                                                         #
#                                                                                 #
###################################################################################
"""

from scipy import constants as cn
import numpy as np
import fqs
import matplotlib.pyplot as plt
import find_beta_corr


def quartic_corr_solver(k_0, gamma_p, rho_z_ic, rho_perp_ic, K_0, I_hat, W, gamma, beta_corr):
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
    :param beta_corr:
    :return:  root_list_final_imag, root_list_final_real ; imaginary and real parts of beta for inputted frequency
    """

    # Form the coefficients of the quartic
    A_q = 1
    B_q = -2 * W
    C_q = W ** 2 - beta_corr ** 2 + I_hat * K_0 ** 2 * rho_perp_ic ** 2 / rho_z_ic ** 3 + 2 * I_hat * K_0 ** 2 / rho_z_ic
    D_q = 2 * W * beta_corr ** 2 - 2 * gamma_p * k_0 * I_hat * K_0 ** 2 / rho_z_ic ** 2 - 2 * W * I_hat * K_0 ** 2 / rho_z_ic
    E_q = - W ** 2 * beta_corr ** 2 - k_0 ** 2 * I_hat * K_0 ** 2 * (
            rho_perp_ic ** 2 / rho_z_ic ** 3) + 2 * k_0 * I_hat * K_0 ** 2 * W * gamma_p / rho_z_ic ** 2

    # Solve the quartic using fqs (can also use python quartic solvers)
    roots = fqs.quartic_roots([A_q, B_q, C_q, D_q, E_q])

    root_list_final_imag = []
    root_list_final_real = []
    omega_list = []

    # Test the roots to find the complex conjugates (should be 2 real roots and 2 complex conjugates, or 4 real)
    a = 4
    for jj in range(4):
        test_root = roots[0][jj]
        if np.imag(test_root) != 0:
            a -= 1
            if np.imag(test_root) < 0:
                a -= 1
                if abs(np.imag(test_root)):
                    root_list_final_imag.append(abs(np.imag(test_root)))
                    root_list_final_real.append(np.real(test_root))

    if a == 4:
        # All real roots - no solutions to the quartic for these frequencies
        root_list_final_imag.append(np.imag(0))
        root_list_final_real.append(np.real(0))

    return root_list_final_imag, root_list_final_real


if __name__ == '__main__':
    # This program, if run as a stand alone, will calculate the quartic and the full system simulation for
    # comparison at several different magnetic field strengths.  This allows the user to fine tune the cyclotron
    # dispersion to maximize the wave amplitude gain. First, define the constant input variables
    gamma_p = np.sqrt(1 + 2 * 1.22 ** 2)
    rho_z_ic = 1.22
    rho_perp_ic = 1.22
    w = 0.05
    K_0 = 14.14          # 1/sqrt(w*h), h = 0.05, used for an average value over z
    I_hat = 0.4          # normalized beam current (multiply by 1356 to get current in amps)

    # First run find_beta_corr.py and select the scenario desired.  Record g, x_c and integer_key
    ell, dispersion_dict_beta_corr = find_beta_corr.find_beta_corr()

    # Enter desired run parameters, xx is the  integer_key for the dispersion_dict
    xx = 17
    g = 0.7631578947368421
    omega_c = 18886924854.0

    # This following section is to help determine the best magnetic field strength for a given x_c, g combination
    Omega_B_ = np.linspace(2.5 * 10 ** 10, 3.5 * 10 ** 10, 10)
    for ii in range(len(Omega_B_)):
        Omega_B = Omega_B_[ii]
        for ii in range(1):
            # Set up the plots for each cyclotron frequency
            fig, (ax1) = plt.subplots(1)

            # Retrieve the data for the run
            phi = list(dispersion_dict_beta_corr.values())[xx][3]
            beta_corr = list(dispersion_dict_beta_corr.values())[xx][4]
            omega_points_ = list(dispersion_dict_beta_corr.values())[xx][1]
            omega_points = [x*cn.c/ell for x in omega_points_]

            beta_corr_list = []
            root_list_imag = []
            root_list_real = []
            omega_list_real = []
            omega_list_imag = []
            for jj in range(len(omega_points)):
                # Select a frequency
                omega = omega_points[jj]

                # Calculate needed variables for the quartic equation
                k_0 = omega / cn.c
                W = (omega * gamma_p - Omega_B) / (cn.c * rho_z_ic)
                gamma = 1 - omega_c ** 2 / omega ** 2
                k_0 = omega/cn.c
                beta_corr_ = beta_corr[jj]

                # We only want beta_corr values that are real, store the roots in a list for plotting
                if np.imag(beta_corr_) == 0:
                    # Solve the quartic
                    root_imag, root_real = quartic_corr_solver(k_0, gamma_p, rho_z_ic, rho_perp_ic, K_0, I_hat, W,
                                                               gamma, np.real(beta_corr_))

                    # Since some beta_corr's do not result in solutions, keep track of which frequencies do
                    if root_imag:
                        root_list_imag.append(root_imag)
                        omega_list_imag.append(omega_points[jj])
                    if root_real:
                        root_list_real.append(root_real)
                        omega_list_real.append(omega_points[jj])

                    beta_corr_list.append(beta_corr_)

            k_z = [x/ell for x in phi]

            # Calculate and plot cyclotron equation
            plt.title(f"Omega_B={Omega_B:.2e}, g = {g:.2e}, cutoff freq = {omega_c:.2e}")
            beta_real = np.linspace(0, 200, 100)
            omega_lin = [x * cn.c * rho_z_ic / gamma_p + Omega_B / gamma_p for x in beta_real]
            ax1.plot(beta_real, omega_lin, 'g.', label='cyclotron')

            # Calculate and plot the dispersion relation for the corrugated waveguide
            ax1.plot(k_z, omega_points, 'b.', label='Dispersion relation')
            omega_c_ = omega_c

            # Plot the roots as a function of wavenumber
            root_imag_neg = [np.nan if x[0] == 0 else abs(x[0]) for x in root_list_imag]
            root_real_neg = [np.nan if x[0] == 0 else abs(x[0]) for x in root_list_real]
            ax1.plot(root_imag_neg, omega_list_imag, 'r+', label=r'$ \beta_{imag}$ from quartic')
            ax1.plot(root_real_neg, omega_list_real, 'm+', label=r'$ \beta_{real}$ from quartic')
            ax1.set_ylabel('Omega [$sec^{-1}$]')
            ax1.set_xlabel(r'$ \beta_{real}$ / $ \beta_{imag}$ [$cm^{-1}]$')
            ax1.set_ylim(0, 5.0*10**10)
            ax1.set_xlim(-5, 150)

            plt.legend()
            plt.show()
