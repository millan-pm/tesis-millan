# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:16:54 2026

@author: milla
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Importación de datos

MA_50_l_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MA_50_1.txt")
MA_90_l_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MA_90_1.txt")
MA_130_l_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MA_130_1.txt")

MB_50_l_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MB_50_1.txt")
MB_90_l_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MB_90_1.txt")
MB_130_l_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MB_130_1.txt")

MC_50_l_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MC_50_1.txt")
MC_90_l_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MC_90_1.txt")
MC_130_l_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MC_130_1.txt")

MD_50_l_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MD_50_1.txt")
MD_90_l_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MD_90_1.txt")
MD_130_l_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MD_130_1.txt")

MA_50_l_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MA_50_1.33.txt")
MA_90_l_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MA_90_1.33.txt")
MA_130_l_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MA_130_1.33.txt")

MB_50_l_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MB_50_1.33.txt")
MB_90_l_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MB_90_1.33.txt")
MB_130_l_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MB_130_1.33.txt")

MC_50_l_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MC_50_1.33.txt")
MC_90_l_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MC_90_1.33.txt")
MC_130_l_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MC_130_1.33.txt")

MD_50_l_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MD_50_1.33.txt")
MD_90_l_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MD_90_1.33.txt")
MD_130_l_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/LINEAL/MD_130_1.33.txt")

MA_50_c_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MA_50_1.txt")
MA_90_c_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MA_90_1.txt")
MA_130_c_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MA_130_1.txt")

MB_50_c_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MB_50_1.txt")
MB_90_c_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MB_90_1.txt")
MB_130_c_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MB_130_1.txt")

MC_50_c_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MC_50_1.txt")
MC_90_c_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MC_90_1.txt")
MC_130_c_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MC_130_1.txt")

MD_50_c_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MD_50_1.txt")
MD_90_c_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MD_90_1.txt")
MD_130_c_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MD_130_1.txt")

MA_50_c_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MA_50_1.33.txt")
MA_90_c_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MA_90_1.33.txt")
MA_130_c_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MA_130_1.33.txt")

MB_50_c_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MB_50_1.33.txt")
MB_90_c_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MB_90_1.33.txt")
MB_130_c_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MB_130_1.33.txt")

MC_50_c_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MC_50_1.33.txt")
MC_90_c_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MC_90_1.33.txt")
MC_130_c_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MC_130_1.33.txt")

MD_50_c_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MD_50_1.33.txt")
MD_90_c_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MD_90_1.33.txt")
MD_130_c_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MD_130_1.33.txt")

MA_ex_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MA_ex_50_1.txt")
MB_ex_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MB_ex_50_1.txt")
MC_ex_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MC_ex_50_1.txt")
MD_ex_1 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MD_ex_50_1.txt")

MA_ex_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MA_ex_50_1.33.txt")
MB_ex_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MB_ex_50_1.33.txt")
MC_ex_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MC_ex_50_1.33.txt")
MD_ex_133 = np.loadtxt("/Users/millanperez/Documents/wip/supporting/CIRCULAR/MD_ex_50_1.33.txt")

#%% Setup cortes 

plt.rcParams.update({'font.size': 18})
plt.rcParams['figure.dpi'] = 150
fs = (16,12)
legend_loc = 0

nwl = 100
nrad= 200
radv = np.linspace(50e-9, 100e-9, nrad)

wl_int = np.linspace(0.45, 0.80, nwl)

RAD, WL = np.meshgrid(radv*1e9, wl_int)

rad_i = 90e-9
irad = np.argmin(np.abs(radv-rad_i))

wl_i = 0.70
iwl = np.argmin(np.abs(wl_int - wl_i))

MA = MA_50_c_1
MB = MB_50_c_1
MC = MC_50_c_1
MD = MD_50_c_1

#"""
MAex = MA_ex_1
MBex = MB_ex_1
MCex = MC_ex_1
MDex = MD_ex_1
#"""

"""
MAex = MA_ex_133
MBex = MB_ex_133
MCex = MC_ex_133
MDex = MD_ex_133
#"""

MA_cutr = MA[irad,:].transpose()
MB_cutr = MB[irad,:].transpose()
MC_cutr = MC[irad,:].transpose()
MD_cutr = MD[irad,:].transpose()

MAex_cutr = MAex[irad,:].transpose()
MBex_cutr = MBex[irad,:].transpose()
MCex_cutr = MCex[irad,:].transpose()
MDex_cutr = MDex[irad,:].transpose()

MA_cutwl = MA[:,iwl].transpose()
MB_cutwl = MB[:,iwl].transpose()
MC_cutwl = MC[:,iwl].transpose()
MD_cutwl = MD[:,iwl].transpose()

MAex_cutwl = MAex[:,iwl].transpose()
MBex_cutwl = MBex[:,iwl].transpose()
MCex_cutwl = MCex[:,iwl].transpose()
MDex_cutwl = MDex[:,iwl].transpose()

#%% wl cuts

plt.figure(figsize=fs)
plt.title(f"$\\lambda = {wl_i}$ $\\mu m$")
plt.plot(radv*1e9, MA_cutwl, '.-', label = "Stokes Matrix")
plt.plot(radv*1e9, MAex_cutwl, label = "Analytical expression")
plt.xlabel("Sphere radius ($n m$)")
plt.ylabel("$|a|^2$")
plt.legend(loc=legend_loc)
plt.show()

plt.figure(figsize=fs)
plt.title(f"$\\lambda = {wl_i}$ $\\mu m$")
plt.plot(radv*1e9, MB_cutwl, '.-', label = "Stokes Matrix")
plt.plot(radv*1e9, MBex_cutwl, label = "Analytical expression")
plt.xlabel("Sphere radius ($n m$)")
plt.ylabel("$|b|^2$")
plt.legend(loc=legend_loc)
plt.show()

plt.figure(figsize=fs)
plt.title(f"$\\lambda = {wl_i}$ $\\mu m$")
plt.plot(radv*1e9, MC_cutwl, '.-', label = "Stokes Matrix")
plt.plot(radv*1e9, MCex_cutwl, label = "Analytical expression")
plt.xlabel("Sphere radius ($n m$)")
plt.ylabel("$Re(ab^*)$")
plt.legend(loc=legend_loc)
plt.show()

plt.figure(figsize=fs)
plt.title(f"$\\lambda = {wl_i}$ $\\mu m$")
plt.plot(radv*1e9, MD_cutwl, '.-', label = "Stokes Matrix")
plt.plot(radv*1e9, MDex_cutwl, label = "Analytical expression")
plt.xlabel("Sphere radius ($n m$)")
plt.ylabel("$Im(ab^*)$$")
plt.legend(loc=legend_loc)
plt.show()

#%% rad cuts

plt.figure(figsize=fs)
plt.title(f"Sphere radius = {round(rad_i*1e9)} $n m$")
plt.plot(wl_int, MA_cutr, '.-', label = "Stokes Matrix")
plt.plot(wl_int, MAex_cutr, label = "Analytical expression")
plt.xlabel("Wavelength ($\mu m$)")
plt.ylabel("$|a|^2$")
plt.legend(loc=legend_loc)
plt.show()

plt.figure(figsize=fs)
plt.title(f"Sphere radius = {round(rad_i*1e9)} $n m$")
plt.plot(wl_int, MB_cutr, '.-', label = "Stokes Matrix")
plt.plot(wl_int, MBex_cutr, label = "Analytical expression")
plt.xlabel("Wavelength ($\mu m$)")
plt.ylabel("$|b|^2$")
plt.legend(loc=legend_loc)
plt.show()

plt.figure(figsize=fs)
plt.title(f"Sphere radius = {round(rad_i*1e9)} $n m$")
plt.plot(wl_int, MC_cutr, '.-', label = "Stokes Matrix")
plt.plot(wl_int, MCex_cutr, label = "Analytical expression")
plt.xlabel("Wavelength ($\mu m$)")
plt.ylabel("$Re(ab^*)$")
plt.legend(loc=legend_loc)
plt.show()

plt.figure(figsize=fs)
plt.title(f"Sphere radius = {round(rad_i*1e9)} $n m$")
plt.plot(wl_int, MD_cutr, '.-', label = "Stokes Matrix")
plt.plot(wl_int, MDex_cutr, label = "Analytical expression")
plt.xlabel("Wavelength ($\mu m$)")
plt.ylabel("$Im(ab^*)$")
plt.legend(loc=legend_loc)
plt.show()

#%% Configuración matplotlib colormaps
plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.dpi'] = 150
fs = (16,12)
padding = 2

cmap_ab = 'viridis'
cmap_reim = 'plasma'
cmap_err = 'inferno'

xticks = [50, 60, 70, 80, 90, 100]
yticks = [0.5, 0.65 , 0.8]

#%% PLOTS 50 1 circular

theta_grados = 50
n2 = 1


nwl = 100
nrad= 200
radv = np.linspace(50e-9, 100e-9, nrad)

wl_int = np.linspace(0.45, 0.80, nwl)

RAD, WL = np.meshgrid(radv*1e9, wl_int)

title = f"a,t={theta_grados},n2={n2},c"
plot_title = f"$|a_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MA_ex_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MA_50_c_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MA_50_c_1.transpose()-MA_ex_1.transpose())/MA_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')
    axi.set_xticks(xticks)
    axi.set_yticks(yticks)
  
plt.savefig(title+".png")
#plt.close()
    
title = f"b,t={theta_grados},n2={n2},c"
plot_title = f"$|b_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MB_ex_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MB_50_c_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MB_50_c_1.transpose()-MB_ex_1.transpose())/MB_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

plt.savefig(title+".png")
plt.close()
    
title = f"reab,t={theta_grados},n2={n2},c"
plot_title = f"$Re(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MC_ex_1.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MC_50_c_1.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MC_50_c_1.transpose()-MC_ex_1.transpose())/MC_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
#plt.close()
    
title = f"imab,t={theta_grados},n2={n2},c"
plot_title = f"$Im(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MD_ex_1.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MD_50_c_1.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MD_50_c_1.transpose()-MD_ex_1.transpose())/MD_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()

    
#%% PLOTS 90 1 circular

theta_grados = 90
n2 = 1


nwl = 100
nrad= 200
radv = np.linspace(50e-9, 100e-9, nrad)

wl_int = np.linspace(0.45, 0.80, nwl)

RAD, WL = np.meshgrid(radv*1e9, wl_int)

title = f"a,t={theta_grados},n2={n2},c"
plot_title = f"$|a_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MA_ex_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MA_90_c_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MA_90_c_1.transpose()-MA_ex_1.transpose())/MA_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')
    
  
plt.savefig(title+".png")
plt.close()
    
title = f"b,t={theta_grados},n2={n2},c"
plot_title = f"$|b_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MB_ex_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MB_90_c_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MB_90_c_1.transpose()-MB_ex_1.transpose())/MB_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"reab,t={theta_grados},n2={n2},c"
plot_title = f"$Re(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MC_ex_1.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MC_90_c_1.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MC_90_c_1.transpose()-MC_ex_1.transpose())/MC_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"imab,t={theta_grados},n2={n2},c"
plot_title = f"$Im(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MD_ex_1.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MD_90_c_1.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MD_90_c_1.transpose()-MD_ex_1.transpose())/MD_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()

#%% PLOTS 130 1 circular

theta_grados = 130
n2 = 1


nwl = 100
nrad= 200
radv = np.linspace(50e-9, 100e-9, nrad)

wl_int = np.linspace(0.45, 0.80, nwl)

RAD, WL = np.meshgrid(radv*1e9, wl_int)

title = f"a,t={theta_grados},n2={n2},c"
plot_title = f"$|a_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MA_ex_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MA_130_c_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MA_130_c_1.transpose()-MA_ex_1.transpose())/MA_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')
    
  
plt.savefig(title+".png")
plt.close()
    
title = f"b,t={theta_grados},n2={n2},c"
plot_title = f"$|b_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MB_ex_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MB_130_c_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MB_130_c_1.transpose()-MB_ex_1.transpose())/MB_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"reab,t={theta_grados},n2={n2},c"
plot_title = f"$Re(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MC_ex_1.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MC_130_c_1.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MC_130_c_1.transpose()-MC_ex_1.transpose())/MC_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"imab,t={theta_grados},n2={n2},c"
plot_title = f"$Im(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MD_ex_1.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MD_130_c_1.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MD_130_c_1.transpose()-MD_ex_1.transpose())/MD_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()    

#%% PLOTS 50 133 circular

theta_grados = 50
n2 = 1.33


nwl = 100
nrad= 200
radv = np.linspace(50e-9, 100e-9, nrad)

wl_int = np.linspace(0.45, 0.80, nwl)

RAD, WL = np.meshgrid(radv*1e9, wl_int)

title = f"a,t={theta_grados},n2={n2},c"
plot_title = f"$|a_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MA_ex_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MA_50_c_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MA_50_c_133.transpose()-MA_ex_133.transpose())/MA_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')
    
  
plt.savefig(title+".png")
plt.close()
    
title = f"b,t={theta_grados},n2={n2},c"
plot_title = f"$|b_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MB_ex_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MB_50_c_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MB_50_c_133.transpose()-MB_ex_133.transpose())/MB_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"reab,t={theta_grados},n2={n2},c"
plot_title = f"$Re(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MC_ex_133.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MC_50_c_133.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MC_50_c_133.transpose()-MC_ex_133.transpose())/MC_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"imab,t={theta_grados},n2={n2},c"
plot_title = f"$Im(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MD_ex_133.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MD_50_c_133.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MD_50_c_133.transpose()-MD_ex_133.transpose())/MD_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
#%% PLOTS 90 133 circular

theta_grados = 90
n2 = 1.33


nwl = 100
nrad= 200
radv = np.linspace(50e-9, 100e-9, nrad)

wl_int = np.linspace(0.45, 0.80, nwl)

RAD, WL = np.meshgrid(radv*1e9, wl_int)

title = f"a,t={theta_grados},n2={n2},c"
plot_title = f"$|a_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MA_ex_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MA_90_c_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MA_90_c_133.transpose()-MA_ex_133.transpose())/MA_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')
    
  
plt.savefig(title+".png")
plt.close()
    
title = f"b,t={theta_grados},n2={n2},c"
plot_title = f"$|b_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MB_ex_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MB_90_c_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MB_90_c_133.transpose()-MB_ex_133.transpose())/MB_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"reab,t={theta_grados},n2={n2},c"
plot_title = f"$Re(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MC_ex_133.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MC_90_c_133.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MC_90_c_133.transpose()-MC_ex_133.transpose())/MC_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"imab,t={theta_grados},n2={n2},c"
plot_title = f"$Im(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MD_ex_133.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MD_90_c_133.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MD_90_c_133.transpose()-MD_ex_133.transpose())/MD_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()

#%% PLOTS 130 133 circular

theta_grados = 130
n2 = 1.33


nwl = 100
nrad= 200
radv = np.linspace(50e-9, 100e-9, nrad)

wl_int = np.linspace(0.45, 0.80, nwl)

RAD, WL = np.meshgrid(radv*1e9, wl_int)

title = f"a,t={theta_grados},n2={n2},c"
plot_title = f"$|a_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MA_ex_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MA_130_c_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MA_130_c_133.transpose()-MA_ex_133.transpose())/MA_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')
    
  
plt.savefig(title+".png")
plt.close()
    
title = f"b,t={theta_grados},n2={n2},c"
plot_title = f"$|b_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MB_ex_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MB_130_c_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MB_130_c_133.transpose()-MB_ex_133.transpose())/MB_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"reab,t={theta_grados},n2={n2},c"
plot_title = f"$Re(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MC_ex_133.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MC_130_c_133.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MC_130_c_133.transpose()-MC_ex_133.transpose())/MC_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"imab,t={theta_grados},n2={n2},c"
plot_title = f"$Im(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MD_ex_133.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MD_130_c_133.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MD_130_c_133.transpose()-MD_ex_133.transpose())/MD_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()

#%% PLOTS 50 1 lineal

theta_grados = 50
n2 = 1


nwl = 100
nrad= 200
radv = np.linspace(50e-9, 100e-9, nrad)

wl_int = np.linspace(0.45, 0.80, nwl)

RAD, WL = np.meshgrid(radv*1e9, wl_int)

title = f"a,t={theta_grados},n2={n2},l"
plot_title = f"$|a_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MA_ex_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MA_50_l_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MA_50_l_1.transpose()-MA_ex_1.transpose())/MA_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')
    
  
plt.savefig(title+".png")
plt.close()
    
title = f"b,t={theta_grados},n2={n2},l"
plot_title = f"$|b_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MB_ex_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MB_50_l_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MB_50_l_1.transpose()-MB_ex_1.transpose())/MB_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"reab,t={theta_grados},n2={n2},l"
plot_title = f"$Re(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MC_ex_1.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MC_50_l_1.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MC_50_l_1.transpose()-MC_ex_1.transpose())/MC_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"imab,t={theta_grados},n2={n2},l"
plot_title = f"$Im(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MD_ex_1.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MD_50_l_1.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MD_50_l_1.transpose()-MD_ex_1.transpose())/MD_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
#%% PLOTS 90 1 lineal

theta_grados = 90
n2 = 1


nwl = 100
nrad= 200
radv = np.linspace(50e-9, 100e-9, nrad)

wl_int = np.linspace(0.45, 0.80, nwl)

RAD, WL = np.meshgrid(radv*1e9, wl_int)

title = f"a,t={theta_grados},n2={n2},l"
plot_title = f"$|a_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MA_ex_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MA_90_l_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MA_90_l_1.transpose()-MA_ex_1.transpose())/MA_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')
    
  
plt.savefig(title+".png")
plt.close()
    
title = f"b,t={theta_grados},n2={n2},l"
plot_title = f"$|b_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MB_ex_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MB_90_l_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MB_90_l_1.transpose()-MB_ex_1.transpose())/MB_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"reab,t={theta_grados},n2={n2},l"
plot_title = f"$Re(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MC_ex_1.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MC_90_l_1.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MC_90_l_1.transpose()-MC_ex_1.transpose())/MC_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"imab,t={theta_grados},n2={n2},l"
plot_title = f"$Im(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MD_ex_1.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MD_90_l_1.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MD_90_l_1.transpose()-MD_ex_1.transpose())/MD_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()

#%% PLOTS 130 1 lineal

theta_grados = 130
n2 = 1


nwl = 100
nrad= 200
radv = np.linspace(50e-9, 100e-9, nrad)

wl_int = np.linspace(0.45, 0.80, nwl)

RAD, WL = np.meshgrid(radv*1e9, wl_int)

title = f"a,t={theta_grados},n2={n2},l"
plot_title = f"$|a_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MA_ex_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MA_130_l_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MA_130_l_1.transpose()-MA_ex_1.transpose())/MA_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')
    
  
plt.savefig(title+".png")
plt.close()
    
title = f"b,t={theta_grados},n2={n2},l"
plot_title = f"$|b_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MB_ex_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MB_130_l_1.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MB_130_l_1.transpose()-MB_ex_1.transpose())/MB_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"reab,t={theta_grados},n2={n2},l"
plot_title = f"$Re(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MC_ex_1.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MC_130_l_1.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MC_130_l_1.transpose()-MC_ex_1.transpose())/MC_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"imab,t={theta_grados},n2={n2},l"
plot_title = f"$Im(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MD_ex_1.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MD_130_l_1.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MD_130_l_1.transpose()-MD_ex_1.transpose())/MD_ex_1.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)


for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()


#%% PLOTS 50 133 lineal

theta_grados = 50
n2 = 1.33


nwl = 100
nrad= 200
radv = np.linspace(50e-9, 100e-9, nrad)

wl_int = np.linspace(0.45, 0.80, nwl)

RAD, WL = np.meshgrid(radv*1e9, wl_int)

title = f"a,t={theta_grados},n2={n2},l"
plot_title = f"$|a_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MA_ex_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MA_50_l_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MA_50_l_133.transpose()-MA_ex_133.transpose())/MA_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')
    
  
plt.savefig(title+".png")
plt.close()
    
title = f"b,t={theta_grados},n2={n2},l"
plot_title = f"$|b_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MB_ex_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MB_50_l_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MB_50_l_133.transpose()-MB_ex_133.transpose())/MB_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"reab,t={theta_grados},n2={n2},l"
plot_title = f"$Re(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MC_ex_133.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MC_50_l_133.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MC_50_l_133.transpose()-MC_ex_133.transpose())/MC_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"imab,t={theta_grados},n2={n2},l"
plot_title = f"$Im(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MD_ex_133.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MD_50_l_133.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MD_50_l_133.transpose()-MD_ex_133.transpose())/MD_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')
    
  
plt.savefig(title+".png")
plt.close()    
    
#%% PLOTS 90 133 lineal

theta_grados = 90
n2 = 1.33


nwl = 100
nrad= 200
radv = np.linspace(50e-9, 100e-9, nrad)

wl_int = np.linspace(0.45, 0.80, nwl)

RAD, WL = np.meshgrid(radv*1e9, wl_int)

title = f"a,t={theta_grados},n2={n2},l"
plot_title = f"$|a_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MA_ex_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MA_90_l_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MA_90_l_133.transpose()-MA_ex_133.transpose())/MA_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')
    
  
plt.savefig(title+".png")
plt.close()
    
title = f"b,t={theta_grados},n2={n2},l"
plot_title = f"$|b_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MB_ex_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MB_90_l_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MB_90_l_133.transpose()-MB_ex_133.transpose())/MB_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"reab,t={theta_grados},n2={n2},l"
plot_title = f"$Re(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MC_ex_133.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MC_90_l_133.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MC_90_l_133.transpose()-MC_ex_133.transpose())/MC_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"imab,t={theta_grados},n2={n2},l"
plot_title = f"$Im(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
#fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MD_ex_133.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MD_90_l_133.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MD_90_l_133.transpose()-MD_ex_133.transpose())/MD_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()

#%% PLOTS 130 133 lineal

theta_grados = 130
n2 = 1.33


nwl = 100
nrad= 200
radv = np.linspace(50e-9, 100e-9, nrad)

wl_int = np.linspace(0.45, 0.80, nwl)

RAD, WL = np.meshgrid(radv*1e9, wl_int)

title = f"a,t={theta_grados},n2={n2},l"
plot_title = f"$|a_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
##fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MA_ex_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MA_130_l_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MA_130_l_133.transpose()-MA_ex_133.transpose())/MA_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')
    
  
plt.savefig(title+".png")
plt.close()
    
title = f"b,t={theta_grados},n2={n2},l"
plot_title = f"$|b_1|^2$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
##fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MB_ex_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MB_130_l_133.transpose(), cmap=cmap_ab, clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MB_130_l_133.transpose()-MB_ex_133.transpose())/MB_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"reab,t={theta_grados},n2={n2},l"
plot_title = f"$Re(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
##fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MC_ex_133.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MC_130_l_133.transpose(), cmap=cmap_reim, clim=(-0.2,0.6))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MC_130_l_133.transpose()-MC_ex_133.transpose())/MC_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
    
title = f"imab,t={theta_grados},n2={n2},l"
plot_title = f"$Im(a_1b_1*)$, $\\theta$={theta_grados}, $n_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
##fig.suptitle(plot_title)

plot00=axs[0].pcolormesh(RAD, WL, MD_ex_133.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Analytical expression')
plot01=axs[1].pcolormesh(RAD, WL, MD_130_l_133.transpose(), cmap=cmap_reim, clim=(-0.3,0.3))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Stokes matrix')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MD_130_l_133.transpose()-MD_ex_133.transpose())/MD_ex_133.max()), cmap=cmap_err)
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Relative error')
plt.tight_layout(pad=padding)

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

  
plt.savefig(title+".png")
plt.close()
        