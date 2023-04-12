# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:10:53 2023

@author: julio
"""

import numpy as np
import matplotlib.pyplot as plt

P_r = 10.64E6 #[W]
C_p_opt = 0.46679535467230016
lam_opt = 8
R = 89.15 #[m]

K_omega = 1/2*1.225*np.pi*R**5*C_p_opt/lam_opt**3
omega_r = (P_r/K_omega)**(1/3)
omega_arr = np.linspace(4*lam_opt/R, 1.5*omega_r,100)
M_gen = np.zeros(len(omega_arr))

for i, omega in enumerate(omega_arr):
    if omega < omega_r:
        M_gen[i] = K_omega*omega**2
    else:
        
        M_gen[i] = P_r/omega
        
plt.figure(dpi=300)
plt.grid()
#plt.title('')
plt.plot(omega_arr, M_gen/1000)
plt.ylabel('$M_G$ [kNm]')
plt.xlabel('$\omega$ [rad/s]')
plt.legend()
plt.figure()

V_0 = np.linspace(4, 24,100)  

P = 1/2*1.225*np.pi*R**2*V_0**3

P = np.where(P < P_r, P, P_r)

V_0_r = (2*P_r/(1.225*np.pi*R**2))**(1/3)

plt.figure(dpi=300)
#plt.title('')

plt.axvline(x= V_0_r, linewidth=1 ,ls='--', color='k')
plt.plot(V_0, P/1E6)
plt.axvline(x= V_0_r, linewidth=1 ,ls='--', color='k')
plt.ylabel('P [MW]')
plt.xlabel('$V_0$ [m/s]')
plt.legend()
plt.figure()
