# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:56:20 2023

@author: julio
"""
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt 
import classes
import functions as f
from scipy import signal

#Inicialize objects
WT = classes.WT()
Wind = classes.Wind()
Config = classes.Config()

if Config.Turbulence:
    Wind.u_turb_tab, Wind.x_turb_lst, Wind.y_turb_lst = f.loadTurb('Turbulence_generator/case2_input.inp', 'Turbulence_generator/case2_sim1.bin')
    

# Initialization
rotor_ang_lst = []
rotor_ang = 0

if Config.Turbulence == False:
    # TotalRev = 2
    # t_end = 2*np.pi / WT.w * TotalRev
    t_end = 310
else:
    t_end = 4*102.4 #[s]

t_arr = np.arange(0,t_end, Config.deltaT)

#Stair
V0_stair_arr = np.linspace(11.4, 25, int(t_end/Config.wvs_interval))
pitch_ang_V0 = np.zeros((len(V0_stair_arr)-1,2))
t_start = 10
t_change = np.zeros(len(pitch_ang_V0))
step = 0
if Config.wind_vel_stair:
    Wind.V_0_H = V0_stair_arr[0]


omega_arr = np.zeros(len(t_arr))
omega_arr[0] = WT.w

M_gen = np.zeros(len(t_arr))

px_lst = [] #Absolute position
py_lst = []

V_y_lst = []
V_z_lst = []

V_0_y_lst = []
V_0_z_lst = []

Wy_lst = []
Wz_lst = []

Wy_old = 0; Wz_old = 0
C_l_lst = []
C_d_lst = []

load_py_elem_lst = []
load_pz_elem_lst = []

blade_norm_load = []

thrust_lst = []
blade_thrust_tab = np.zeros((WT.B, len(t_arr)))
power_m_arr = np.zeros(len(t_arr))
power_e_arr = np.zeros(len(t_arr))

#Control
K_omega = 1/2*Wind.rho*np.pi*WT.R**5*WT.C_p_opt/WT.lam_opt**3
omega_r = (WT.P_r/K_omega)**(1/3)
omega_ref = 1.0*omega_r
theta_setpoint = np.zeros(len(t_arr)) 
theta_P = np.zeros(len(t_arr)) 
theta_I = np.zeros(len(t_arr))  
theta_max = np.radians(45)
theta_min = np.radians(0)
Kp = 1.5 #[rad/(rad/s)]
KI = 0.64 #[rad/rad]
K1 = np.radians(14) #degrees
theta_dot_max = np.radians(10) #[degrees/s]



a_arr = np.zeros((len(t_arr), WT.B, len(WT.r_lst)))

for t_step, t in enumerate(t_arr):
    #Wind velocity stair test
    if Config.wind_vel_stair:
        if t-t_start > Config.wvs_interval:
            t_change[step] = t
            t_start = t
            step = step + 1
            pitch_ang_V0[step-1,:] = [V0_stair_arr[step-1], WT.pitch_ang]
            Wind.V_0_H = V0_stair_arr[step]
             
    
    #Inicialize blade loop
    thrust = 0
    M_aero = 0
    for i, b in enumerate(range(WT.B)):
        WT.blade = b
        
        #Update position of blade
        blade_ang = rotor_ang + b*2*np.pi/WT.B
        
        #Inicialize radius loop
       
        load_py_lst = []
        load_pz_lst = []
        for element in range(len(WT.r_lst)-1):
            WT.element = element
            
            #Calculate blade element position
            pos = f.get_position(WT, blade_ang, element)
            
            #V_0 in the blade element
            V_0_vec = f.get_v0 (WT, Wind, Config, pos, t_step)
            
            #Quasy Steady Induced Wind
            Wy_qs, Wz_qs, a, py, pz, C_l, C_d = f.QuasySteadyIW(WT, Config, Wind, V_0_vec, WT.last_W[0,element,b], WT.last_W[1,element,b], blade_ang, element)
            a_arr[t_step, b, element] = a
            
            #Dynamic Filtering
            if Config.DynFilter:
                Wy_new, Wz_new = f.DynFiltering (Wy_qs, Wz_qs, a, LA.norm(V_0_vec), element, b, WT, Config)
            else:
                Wy_new, Wz_new = Wy_qs, Wz_qs
            
            WT.last_W_qs[0,element,b] = Wy_qs
            WT.last_W_qs[1,element,b] = Wz_qs
            WT.last_W[0,element,b] = Wy_new
            WT.last_W[1,element,b] = Wz_new
                       
            load_py_lst.append(py)
            load_pz_lst.append(pz)
            
            #Saving info of 9th element of the 1st blade
            if element == 8 and b == 0:
                px_lst.append(pos[0])
                py_lst.append(pos[1])
                V_y_lst.append(V_0_vec[1])
                V_z_lst.append(V_0_vec[2])
                C_l_lst.append(C_l)
                C_d_lst.append(C_d)
                load_py_elem_lst.append(py)
                load_pz_elem_lst.append(pz)
                Wy_lst.append(Wy_new)
                Wz_lst.append(Wz_new)
            
        #Last element has no loads
        load_py_lst.append(0)
        load_pz_lst.append(0)
        
        blade_thrust_tab[WT.blade, t_step] = np.trapz(load_pz_lst, WT.r_lst)
        thrust += np.trapz(load_pz_lst, WT.r_lst) #Add each blade thrust
        py_times_r = [load_py_lst[i]*WT.r_lst[i] for i in range(len(WT.r_lst))]
        M_aero += np.trapz(py_times_r, WT.r_lst)
        
    thrust_lst.append(thrust)
    
    power_m_arr[t_step] = M_aero*WT.w
    
    #Update of rotor position
    rotor_ang += WT.w*Config.deltaT
    rotor_ang_lst.append(rotor_ang)
    
    
    
    
    
    
    
    #Control
    if Config.pitch_control:
        
        #Torque control for above and below rated wind speeds
        if omega_arr[t_step] < omega_r:
            M_gen[t_step] = K_omega*omega_arr[t_step]**2
        else:
            M_gen[t_step] = WT.P_r/omega_arr[t_step]
        
        power_e_arr[t_step] = WT.w * M_gen[t_step]
        
        #Fixed generator torque, variable pitch angle
        
        if t_step < len(t_arr)-1: #Skips the last time step
            #Pitch control
            GK = 1/(1 + WT.pitch_ang/K1)
            theta_P[t_step+1] = GK*Kp*(WT.w - omega_ref)
            theta_I[t_step+1] = theta_I[t_step] + GK*KI*(WT.w - omega_ref)*Config.deltaT
            theta_setpoint [t_step+1] = theta_P[t_step+1] + theta_I[t_step+1]
            
            #Checking ang. velocity
            if theta_setpoint [t_step+1] > WT.pitch_ang + theta_dot_max*Config.deltaT:
                theta_setpoint [t_step+1] = WT.pitch_ang + theta_dot_max*Config.deltaT
            elif theta_setpoint [t_step+1] < WT.pitch_ang - theta_dot_max*Config.deltaT:
                theta_setpoint [t_step+1] = WT.pitch_ang - theta_dot_max*Config.deltaT
                
            #Checking actuator pos.
            if theta_setpoint [t_step+1] >= theta_max:
                theta_setpoint [t_step+1] = theta_max
                theta_I[t_step+1] = theta_I[t_step]
            elif theta_setpoint [t_step+1] < theta_min:
                theta_setpoint [t_step+1] = theta_min
                theta_I[t_step+1] = theta_I[t_step]
            
            WT.pitch_ang = theta_setpoint[t_step+1]
                
            #Rotor Dynamics
            omega_arr[t_step+1] = omega_arr[t_step] + (M_aero - M_gen[t_step])/WT.I_r*Config.deltaT
            WT.w = omega_arr[t_step+1]
        

#Plot

#Q2

# plt.figure(dpi=300)
# plt.plot(t_arr, omega_arr)
# omega_opt =Wind.V_0_H*WT.lam_opt/WT.R
# plt.axhline(y= omega_opt, linewidth=1 ,ls='--', color='k')
# plt.ylabel('$\omega$ [rad/s]')
# plt.xlabel('Time [s]')
# plt.figure()

# Cp = power_e_arr/(1/2*1.225*np.pi*WT.R**2*Wind.V_0_H**3)
# plt.figure(dpi=300)
# plt.plot(t_arr[2:], Cp[2:])
# plt.axhline(y= WT.C_p_opt, linewidth=1 ,ls='--', color='k')
# plt.text(100, WT.C_p_opt-0.005, f'$C_p = {WT.C_p_opt:.3f}$', horizontalalignment = 'right')
# plt.ylabel('$C_p$ [-]')
# plt.xlabel('Time [s]')
# plt.figure()

# warmup = 10
# plt.figure(dpi=300)
# #plt.title('Power')
# plt.plot(t_arr[warmup:],power_m_arr[warmup:]/1E6,label='Mechanical Power')
# plt.plot(t_arr[warmup:], power_e_arr[warmup:]/1E6,label='Electrical Power')
# plt.ylabel('Power [MW]')
# plt.xlabel('Time [s]')
# plt.legend()
# plt.figure()

# plt.figure(dpi=300)
# plt.grid()
# plt.plot([4]+list(pitch_ang_V0[:,0]),[0]+list(np.degrees(pitch_ang_V0[:,1])))
# plt.ylabel('Pitch [$\circ$]')
# plt.xlabel('$V_0$ [m/s]')
# plt.figure()


# ig, ax1 = plt.subplots(dpi=300)
# ax2 = ax1.twinx()
# ax1.plot(t_arr, np.degrees(theta_setpoint))
# ax2.step(t_change, pitch_ang_V0[:,0],color='tab:red', where = 'pre')
# ax1.set_xlabel("Time [s]")
# ax1.set_ylabel("Pitch [$\circ$]", color='tab:blue')
# ax2.set_ylabel("$V_0$ [m/s]", color='tab:red')

#Q3

# Cp = power_e_arr/(1/2*1.225*np.pi*WT.R**2*Wind.V_0_H**3)
# plt.figure(dpi=300)
# plt.plot(t_arr[2:], Cp[2:])
# plt.axhline(y= WT.C_p_opt, linewidth=1 ,ls='--', color='k')
# plt.text(100, WT.C_p_opt-0.005, f'$C_p = {WT.C_p_opt:.3f}$', horizontalalignment = 'right')
# plt.ylabel('$C_p$ [-]')
# plt.xlabel('Time [s]')
# plt.figure()

# plt.figure(dpi=300)
# plt.plot(t_arr, omega_arr)
# omega_opt =Wind.V_0_H*WT.lam_opt/WT.R
# plt.axhline(y= omega_r, linewidth=1 ,ls='--', color='k')
# plt.ylabel('$\omega$ [rad/s]')
# plt.xlabel('Time [s]')
# plt.figure()

# plt.figure(dpi=300)
# #plt.title('$M_G$')
# plt.plot(t_arr, M_gen/1000)
# plt.ylabel('Torque [kNm]')
# plt.xlabel('Time [s]')
# plt.figure()

# plt.figure(dpi=300)
# #plt.title('$ \{theta} _p$')
# plt.plot(t_arr, np.degrees(theta_setpoint))
# plt.ylabel('Pitch [$^\circ$]')
# plt.xlabel('Time [s]')
# plt.figure()

# warmup = 10
# plt.figure(dpi=300)
# #plt.title('Power')
# plt.plot(t_arr[warmup:],power_m_arr[warmup:]/1E6,label='Mechanical Power')
# plt.plot(t_arr[warmup:], power_e_arr[warmup:]/1E6,label='Electrical Power')
# plt.ylabel('Power [MW]')
# plt.xlabel('Time [s]')
# plt.legend()
# plt.figure()

# plt.figure(dpi=300)
# plt.title('KP & KI')
# plt.plot(t_arr, theta_I)
# plt.plot(t_arr, theta_P)
# plt.ylabel('')
# plt.figure()

# plt.figure(dpi=300)
# #plt.title('KP & KI')
# plt.plot(t_arr, np.array(thrust_lst)/1E3)
# plt.ylabel('Thrust [kN]')
# plt.xlabel('Time [s]')
# plt.figure()

# print('Cp = ', power_m_arr[t_step]/(1/2*1.225*np.pi*WT.R**2*Wind.V_0_H**3))
