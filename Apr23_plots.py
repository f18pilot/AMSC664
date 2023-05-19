#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 3 15:30:00 2023

@author: bscheufele

###################################################################################
#   Project: Using Periodic Structure in Waveguide to amplify RF signal           #
#   Program: Plotting the simulation results and calculating the power slope      #
#   Author:  B Scheufele                                                          #
#                                                                                 #
#   Version: 3.0 (updated 3 Apr 2023)                                            #
#                                                                                 #
#   Dependencies: scipy, numpy, cmath, matplotlib                                 #
#                                                                                 #
#   Approach: The methods below are described by the paper "AMSC664 Research      #
#             Final Report May 2023".    The power curve is plotted and the       #
#             slope of the curve is calculated by restricting the linear          #
#             least squares calculation to the range of values where it is        #
#             linear.  The algorithm used is 'curve_fit' from scipy.optimize.     #
#                                                                                 #
###################################################################################
"""
from scipy import constants as cn
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import cmath as cm


def f(z, A, B):
    """
    Define a function for linear least squares fit
         y = A * exp(B*z)
    :param z: axial coordinates
    :param A: constant
    :param B: constant
    :return:  function value, y
    """
    return A * np.exp(B*z)


def plot_wave(particle_dict, z, steps, N, jj, omega_points):
    """
    This function takes the values from the particle dictionary and plots vs z

    :param particle_dict:
    :param z:
    :param steps:
    :param N:
    :param jj:
    :param omega_points:
    :return:
    """
    # Plot the results for I, V, gamma_p, power
    omega = omega_points[jj]
    I_0 = []
    V_0 = []
    P_wave = []
    #print('inside plot', "inside plot")

    for num1, num2 in zip((particle_dict.get(f"I_0_{jj}")), particle_dict.get(f"V_0_{jj}")):
        I_0.append(num1)
        V_0.append(num2)
    P_w = [0.5 * (np.conj(y) * x + y * np.conj(x)) for x, y in zip(I_0[:-1], V_0[:-1])]
    Conj_V = [np.conj(x) for x in V_0[:-1]]
    fig, (ax2) = plt.subplots(1)
    omega_GHz = omega/(2*np.pi)
    fig.suptitle(f'Power vs z,  omega = {omega_GHz:.2e} GHz')
    V_real = [((cn.m_e*cn.c**2)/cn.elementary_charge) * np.real(x) for x in V_0[:-1]]
    I_imag = [((cn.m_e*cn.c**2)/cn.elementary_charge) * (np.sqrt(cn.epsilon_0/cn.mu_0)) * np.imag(x) for x in I_0[:-1]]
    P_w_VI = [x*y for x, y in zip(V_real, I_imag)]
    #ax1.plot(z, V_0[:-1], label='V_real for pts 500:1000')
    # ax1.plot(z[0:100], V_real[0:100], label='V_real for pts 250:750')
    # for zc in z[0:100]:
    #     ax1.axvline(x=zc)
    #plt.axvline(x=z_lines, linewidth=0.5, color="gray")

    # ax1.plot(z[500:1000], I_imag[500:1000], label='I_imag for pts 500:1000')
    P_watts = [((cn.m_e**2*cn.c**4)/cn.elementary_charge**2 * (np.sqrt(cn.epsilon_0/cn.mu_0))) * x for x in P_w]
    #print('e/m', np.sqrt(cn.epsilon_0/cn.mu_0))
    #print('V', V_real[500:1000])
    #print('I', I_imag[500:1000])
    conv_factor = ((cn.m_e**2*cn.c**4)/cn.elementary_charge**2 * (np.sqrt(cn.epsilon_0/cn.mu_0) ))
    P_watt = [conv_factor * abs(x) for x in P_w]
    #print('P', np.real(P_w[500:1000]))
    #print(P_watt)
    #print('z', z)
    ax2.plot(z, P_watt)
    #ax2.axvline(x=0.5)

    #print(z[100])
    #ax2.plot(z, P_watts, label='P_w')
    popt, pcov = curve_fit(f, z[0:50], (P_watts[0:50]))  # your data x, y to fit
    y = [popt[0]*np.exp(popt[1]*x) for x in z ]
    print('slope, intercept', popt[0], popt[1])
    ax2.plot(z, y, label=f'slope = {popt[1]:.2e}')
    ax2.set_yscale("log")
    ax2.set_ylim(1, 10**9)
    ax2.set_xlim(0, 1)
    print('factor to convert to volt', cn.m_e*cn.c**2/cn.elementary_charge)
    print('factor to convert to ampere', cn.m_e * cn.c ** 2 / cn.elementary_charge * (np.sqrt(cn.epsilon_0/cn.mu_0)))
    factor_power = ((cn.m_e**2*cn.c**4)/cn.elementary_charge**2) * (np.sqrt(cn.epsilon_0/cn.mu_0))
    print('VI', f'{factor_power:.3e}')
    plt.legend()
    ax2.set_xlabel('z [m]')
    #ax1.set_ylabel('V [V]')
    #ax1.set_xlim(0, 1)
    ax2.set_ylabel('Power [W]')
    #ax2.set_xlim(0, 1)
    plt.show()

    return V_0, I_0, P_w

def plot_beam(particle_dict, z, steps, N, jj, omega_points, I_b):
    """

    :param particle_dict:
    :param z:
    :param steps:
    :param N:
    :param jj:
    :param omega_points:
    :param I_b:
    :return:
    """
    sum_gamma_P = np.zeros(steps)
    sum_g = []
    if N == 0:
        P_b = np.zeros(steps)
    else:
        for ll in range(N):
            rho_perp = []
            rho_z = []
            psi = []
            # print('particle number', ll)
            for num_1, num_2 in zip((particle_dict.get(f"p_perp{jj}{ll}")), particle_dict.get(f"p_z{jj}{ll}")):
                rho_perp.append(num_1)
                rho_z.append(num_2)

            for num_3 in particle_dict.get(f"psi{jj}{ll}"):
                psi.append(num_3)
            # for num_p3 in particle_dict.get(f"p_perp{jj}{ll}")
            #     psi.append(num_p3)
            #print('rho_p', rho_perp)
            #print('rho_z', rho_z)
            gamma_P = [np.sqrt(1 + x ** 2 + y ** 2) for x, y in zip(rho_perp[:-1], rho_z[:-1])]
            #print('size gamma_p', gamma_P)
            new_psi = [np.real(np.exp(x * 1j)) for x in psi]
            # print('factor in psi', (-1j * np.sqrt(1/(2*w*h_0))))
            plt.figure(1)
            # fig, (ax1, ax2, ax3) = plt.subplots(3)
            # plt.figure(1)
            plt.plot(z, rho_perp[:-1], '.', label=f'rho_p particle {ll:.1e}')
            plt.legend()
            # ax4.plot(z, rho_perp, label=f'rho_p particle {ll:.2e}')
            plt.figure(2)
            plt.plot(z, psi[:-1], label=f'psi {ll:.2e}')
            plt.legend()
            # ax6.plot(z[:-1], dGdz, label='Ibeam_dG/dz')
            plt.figure(3)
            plt.plot(z, rho_z[:-1], label=f'rho_z {ll:.2e}')
            plt.legend()
            # ax4.plot(z, rho_perp, label=f'rho_p particle {ll:.2e}')
            sum_gamma_P = [x + y for x, y in zip(sum_gamma_P, gamma_P)]
        # print('sum_gamma_P', sum_gamma_P)


        #print('sum_gamma_P', sum_gamma_P)
        ave_gamma_P = [x / N for x in sum_gamma_P]
        check = [x - y for x, y in zip(sum_gamma_P, ave_gamma_P)]
        #print('check', check)
        #K_0 = 14.14213562373095
        #test_factor = [x / y for x, y in zip(rho_perp, rho_z)]  # testing line 25 in write-up
        # test_factor_KV = [K_0 * x * np.conj(y) for x, y in zip(test_factor, V_0)]
        #print('size ave_gamma_p', np.shape(ave_gamma_P))
        P_b = [I_b * x for x in sum_gamma_P]
        #print('P_b', P_b)
        #checksum_gamma_P = [np.conj(x) * y for x, y in zip(V_0, new_ST0)]
        dG = np.gradient(ave_gamma_P, z)
        dP_b = np.gradient(P_b)
        I_dG = [I_b * x for x in dG]
    plt.show()
    return P_b, I_dG, dP_b


def plot_STO(particle_dict, ST0, zpoints, step, N, jj, omega_points, V_0, I_0, I_b):
    """

    :param particle_dict:
    :param ST0:
    :param zpoints:
    :param step:
    :param N:
    :param jj:
    :param omega_points:
    :param V_0:
    :param I_0:
    :param I_b:
    :return:
    """
    number_freq = 1
    omega = omega_points[jj]
    z = zpoints
    new_ST0 = ST0[::4]
    #print('size st0', np.shape(new_ST0))

    for kk in range(number_freq):
        ST0_freq = new_ST0[kk * number_freq: kk * number_freq + step: 1]
        ST0_abs = []
        ST0_phase = []
        for ii in range(step):
            ST0_abs.append(abs(ST0_freq[ii]))
            ST0_phase.append(cm.phase(ST0_freq[ii]))
        # print('ST0', ST0_abs)
        fig, (ax8) = plt.subplots(1)
        fig.suptitle(f'Gamma_p, S_T0, omega = {omega:.2e}')

        # print('power_total', P_total)
        # ax7.plot(z, P_b, '.', label='P_beam')
        ST0_ABS = [abs(x) for x in new_ST0]
        # ax7.plot(z, test_factor_KV, label='rho_p/rho_z*KV')
        # ax7.plot(z[:-1], dG, label='dG')
        # ax7.plot(z[:-1], dP_b, label='dP_b')
        # ax7.plot(z[:-1], ST0_ABS, label='abs(S_T0')
        #ax8.plot(z[:-1], I_dG, label='gI*dGdz')
        ax8.plot(z, new_ST0, label='S_T0 ')
        # ax7.set_xlabel('z in meters')
        ax8.set_xlabel('z in meters')
        # ax7.legend()

        # ax7.set_ylabel('Gamma_p')
        ax8.set_ylabel('S_T0')
        ax8.legend()
        plt.show()
        return

def power(P_w, P_b, P_total, dPdz, omega_points, jj, zpoints):
    """

    :param P_w:
    :param P_b:
    :param P_total:
    :param dPdz:
    :param omega_points:
    :param jj:
    :param zpoints:
    :return:
    """
    omega = omega_points[jj]
    z = zpoints
    print('zpoints', z)
    fig, (ax8, ax9, ax10) = plt.subplots(3)
    fig.suptitle(f'Power, omega = {omega:.2e}')
    P_total = [x + y for x, y in zip(P_b, P_w)]
    #ST0_ABS = [abs(x) for x in new_ST0]
    # print('power_total', P_total)
    ax8.plot(z, P_b, '.', label='log P_beam')
    ax9.plot(z, P_w, '.', label='P_wave')
    ax10.plot(z, P_w, '.', label='log P_wave')
    ax10.set_yscale("log")
    ax10.set_xlabel('z in meters')
    ax8.set_ylabel('P_b')
    ax9.set_ylabel('P_w')
    ax10.set_ylabel('log P_w')
    ax8.legend()
    ax9.legend()
    ax10.legend()
    plt.show()
    return

def energy_con(new_ST0, I_dG, STO_star_V0, P_b, P_w, omega_points, jj, zpoints):
    """

    :param new_ST0:
    :param I_dG:
    :param STO_star_V0:
    :param P_b:
    :param P_w:
    :param omega_points:
    :param jj:
    :param zpoints:
    :return:
    """
    # plotting STO power conservation
    omega = omega_points[jj]
    z = zpoints
    # calculate the derivative of the power of the wave
    dP = np.gradient(P_w)
    dz = np.gradient(zpoints)
    dPdz = dP / dz
    fig, (ax11, ax12, ax13, ax14) = plt.subplots(4)
    fig.suptitle(f'Power Conservation Equations, omega = {omega:.2e}')
    print('HERE!')
    # blue, orange, green

    ax11.plot(z, I_dG, 'b', label='dG')
    ax12.plot(z, STO_star_V0, color='orange', label='STO*V_0')
    ax13.plot(z, dPdz, 'g', label='dPdz')
    ax14.plot(z, I_dG, label='I*dGdz')
    ax14.plot(z, STO_star_V0,  label='S_T0 V*')
    ax14.plot(z, -0.5*dPdz, label='-0.5*dPdz')
    ax11.set_ylabel('I_dG')
    ax12.set_ylabel('STO*V_0')
    ax13.set_ylabel('dPdz')
    ax14.set_ylabel('All three')
    ax14.set_xlabel('z in meters')
    # ax7.legend()
    ax11.legend()
    ax12.legend()
    ax13.legend()
    ax14.legend()
    # ax7.set_ylabel('Gamma_p')
    ax11.set_ylabel('S_T0')
    plt.show()
    return

def quartic(roots_list, rho_z_ic, gamma_p, Omega_B, omega_c, omega_points):
    """

    :param roots_list:
    :param rho_z_ic:
    :param gamma_p:
    :param Omega_B:
    :param omega_c:
    :param omega_points:
    :return:
    """

    N = len(roots_list)

    fig, (ax2) = plt.subplots(1)
    fig.suptitle(f'roots real and imag vs omega, omega =1.85*10^10 to 3.8*10^10 , {N} points')

    beta_real = np.linspace(0, 150, 500)
    omega_lin = [x * cn.c * rho_z_ic / gamma_p + Omega_B / gamma_p for x in beta_real]
    ax2.plot(beta_real, omega_lin, 'g', label='cyclotron')

    omega_dispersion = [np.sqrt(omega_c ** 2 + cn.c ** 2 * x ** 2) for x in beta_real]
    ax2.plot(beta_real, omega_dispersion, 'b', label='Dispersion relation')
    ax2.plot(-10*np.imag(roots_list), omega_points, 'm', label='10X beta_imag from quartic')
    ax2.plot(np.real(roots_list), omega_points, 'r', label='beta real from quartic')

    ax2.set_ylabel('Omega')
    ax2.set_xlabel('beta_real/beta_imag')


    ax2.plot(0, omega_c, 'r.', label=f'cutoff freq {omega_c:.1e}')
    plt.legend()
    plt.show()

    return
