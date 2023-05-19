#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Mar 7 16:31:00 2023

@author: bscheufele

#####################################################################################
#   Project: Using Periodic Structure in Waveguide to amplify RF signal             #
#   Program: Finding beta_g vs phi curves for values of g, x_c, and determining     #
#            beta_corr values with correlated omegas for specific g and x_c values  #
#   Author:  B Scheufele                                                            #
#                                                                                   #
#   Version: 3.0 (updated 3 Mar 2013)                                               #
#                                                                                   #
#   Dependencies: Scipy, numpy, matplotlib, cmath                                   #
#                                                                                   #
#   Approach: Following the approach in overleaf "Quartic solution for              #
#             Corrugated Waveguide Part II", values of g, x_c, and 1/L are          #
#             chosen, and x_w is solved for a range of theta (0, ~2*pi);            #
#               where     x_w = sqrt(theta^2 + x_c^2 + g^2)                         #
#             For each theta a value of phi is calculated, where                    #
#               cos(phi) = cos(theta) + g^2/theta^2[ cos(theta) -1]                 #
#             Next the slope of the dispersion relation d x_w/d phi is              #
#             calculated, and curves that have their max values in                  #
#             range (0.67c-0.72c) are selected and saved in "dispersion_dict"       #
#                                                                                   #
#####################################################################################
"""
from scipy import constants as cn
import numpy as np
import matplotlib.pyplot as plt
import cmath as cm


def find_beta_corr():
    # Number of frequencies
    N = 20

    # Select period of structure and geometry of the structure
    ell = 1 / 35
    g_list = np.linspace(0.01, 1.6, N)

    # Initialize the dictionary
    dispersion_dict = {}
    integer_key = 0

    # Calculate x_w and phi, determine the max value of d x_w / d phi
    x_c = np.linspace(0.2, 4.0, N)
    theta = np.linspace(0.001, 6.2, 5 * N)  # Avoid divide by zero
    # fig, (ax1) = plt.subplots(1)
    for jj in range(len(g_list)):
        g = g_list[jj]

        for ii in range(len(x_c)):
            x_c_ = x_c[ii]
            x_w = [np.sqrt(x ** 2 + x_c_ ** 2 + g ** 2) for x in theta]
            cos_phi = [np.cos(x) + (g ** 2 / x ** 2) * (np.cos(x) - 1) for x in theta]
            phi = [cm.acos(x) for x in cos_phi]

            # Select g values from a range (default to full range), plot to review dispersion curve
            if 0.01 <= g and g <= 1.6:
                k_z = [x/ell for x in phi]
                real_k_z = [np.real(x) for x in k_z]
                imag_k_z = [-1*np.imag(x) for x in k_z]
                # ax1.plot(real_k_z, x_w, 'b+', label=fr"x_c = {x_c_:.2e}")
                # ax1.plot(imag_k_z, x_w, 'm+', label=fr"x_c = {x_c_:.2e}")
                # ax1.set_xlabel(r'$\phi \quad in \quad radians$')
                # ax1.set_ylabel(r'$\omega [radians/sec]$')
                # ax1.set_title(fr'$x_w \quad vs\quad  \phi \quad  for\quad g = {g_list[jj]:.2f}$ and various x_c={x_c_:.2e} ')
                # # ax1.legend()

            dx_w_d_phi = np.gradient(x_w, phi)        # d x_w / d phi is the slope of the dispersion curve
            beta_g = dx_w_d_phi

            # Find curves with max value 0.67c - 0.72c by first removing values of beta_g which are 'NaN' or if v > c
            cleaned_beta_list1 = [x for x in beta_g if ~np.isnan(x)]
            cleaned_beta_list = [x for x in cleaned_beta_list1 if x < 1.0]

            # Next, find the maximum beta_g value for this curve
            max_y = np.max(cleaned_beta_list)

            index = cleaned_beta_list.index(max_y)
            max_x = phi[index]
            max_z = x_w[index]
            beta_corr = [x / ell for x in phi]

            if max_y <= 0.67 or max_y >= 0.72:
                pass
            else:
                dispersion_dict[integer_key] = ((g, x_c_, max_z, max_y, max_x), x_w, beta_g, phi, beta_corr)

                integer_key += 1
    # plt.show()
    return ell, dispersion_dict


if __name__ == '__main__':
    # Calculate curves near cyclotron slope (0.7c): to display plots of curves, uncomment in function
    ell, dispersion_dict = find_beta_corr()

    # To select a specific g or cutoff frequency omega_c, take from list printed, the number gives
    # the integer key to the dispersion_dict

    for ii in range(len(dispersion_dict)):
        print(ii)
        print('g', list(dispersion_dict.values())[ii][0][0])
        print('omega_c', (cn.c/ell)*list(dispersion_dict.values())[ii][0][1])

