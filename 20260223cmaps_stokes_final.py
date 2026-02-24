"""
Created on Thu Jan 29 11:00:03 2026

@author: milla

Calcula los campos E y H dispersados por una esfera de índice n2 y size parameter q=ka
"""

import numpy as np
from scipy.special import sph_harm, spherical_jn, spherical_yn, sph_harm_y, assoc_legendre_p
import matplotlib.pyplot as plt
from math import factorial as f

def dY_theta(l, m, Theta, Phi): 
    Y_p = sph_harm_y(l, m+1, Theta, Phi)
    Y_n = sph_harm_y(l, m-1, Theta, Phi)

    
    c_p = np.sqrt((l - m) * (l + m + 1))
    c_n = np.sqrt((l + m) * (l - m + 1))
   
    dY = 0.5 * (
        c_p * Y_p * np.exp(-1j * Phi) 
        - c_n * Y_n * np.exp(1j * Phi)
    )
  
    return dY

def X_lm(l, m, Theta, Phi):
    Y = sph_harm_y(l, m, Theta, Phi)
    dY_t = dY_theta(l, m, Theta, Phi)

    pre = 1/np.sqrt(l*(l+1))
    
    X_theta = pre * (1*m/np.sin(Theta)) * Y # ORIGINAL MIllan
    X_theta = pre * (-1*m/np.sin(Theta)) * Y # Corregido
    X_phi   = pre * -1j * dY_t

    return X_theta, X_phi

def z_l(l, kr, forma='j'):
    if forma == 'j':
        return spherical_jn(l, kr+0j)
    if forma == 'h':
        return spherical_jn(l, kr) + 1j * spherical_yn(l, kr)
    
def dz_l(l, kr, forma='j'):
    if forma == 'j':
        return spherical_jn(l, kr+0j, True)
    if forma == 'h':
        return spherical_jn(l, kr, True) + 1j * spherical_yn(l, kr, True)

def M_lm(l, m, k, R, Theta, Phi, forma='j'):
    kr = k*R
    z = z_l(l, kr, forma)

    X_theta, X_phi = X_lm(l, m, Theta, Phi)

    M_r     = 0j
    M_theta = z * X_theta
    M_phi   = z * X_phi

    return M_r, M_theta, M_phi

def N_lm(l, m, k, R, Theta, Phi, forma='j'):
    kr = k*R
    z  = z_l(l, kr, forma)
    dz = dz_l(l, kr, forma)

    Y = sph_harm_y(l, m, Theta, Phi)

    X_theta, X_phi = X_lm(l, m, Theta, Phi)
 
    N_r =  z * 1/kr * np.sqrt(l*(l+1)) * Y # ORIGINAL Millan
    N_r =  -z * 1/kr * np.sqrt(l*(l+1)) * Y # Corregido
    N_theta = -1j/kr * X_phi * (z + kr * dz)
    N_phi   = 1j/kr * X_theta * (z + kr * dz)

    return N_r, N_theta, N_phi

def BSC_plana(l, px, py):
    common = np.sqrt(np.pi*(2*l+1)) * 1j**(l) 
    # Originales Neves
    G_TM_p = common * (-1j*px-py) 
    G_TM_m = common * (+1j*px-py) 
    G_TE_p = common * (px - 1j*py)
    G_TE_m = common * (px + 1j*py)
    return G_TM_p, G_TM_m, G_TE_p, G_TE_m

def sp_hankel_1(l, z):
    """
    sp_hankel_1(l, z)
    Función de hankel esférica de primera especie y su derivada.
    h1_l(z) = j_n(z) + iy_l(z)
    h1'_l(z) = j'_n(z) + iy'_l(z)

    Depende de scipy.special

    l: orden de la función (entero)
    z: variable sobre la que se opera (real)

    Devuelve un vector con el valor de la funcion y su primera derivada. (complejo)
    """
    normal = spherical_jn(l, z) + 1j*spherical_yn(l, z)
    derivada = spherical_jn(l, z, True) + 1j*spherical_yn(l, z, True)
    return normal, derivada

def riccati_bessel(l, z):
    """
    ricatti_bessel(l, z)
    Función de Riccati-Bessel y su derivada.
    psi_l(z) = z*j_l(z)
    psi'_l(z)= j_l(z) + z * j'_l(z)

    Depende de scipy.special

    l: orden de la función (entero)
    z: variable sobre la que se opera (real)

    Devuelve un vector con el valor de la funcion y su primera derivada. (complejo)
    """
    normal = z * spherical_jn(l, z)
    derivada = spherical_jn(l, z) + z * spherical_jn(l, z, True)
    return normal, derivada

def riccati_hankel_1(l, z):
    """
    riccati_hankel_1(l, z)
    Función de Riccati-Hankel y su derivada.
    xi_l(z) = z*h1_l(z)
    xi'_l(z) = h1_l(z) + z*h1'_l(z)

    Depende de sp_hankel y scipy.special

    l: orden de la función (entero)
    z: variable sobre la que se opera (real)

    Devuelve un vector con el valor de la funcion y su primera derivada. (complejo)
    """
    normal = z * sp_hankel_1(l, z)[0]
    derivada = sp_hankel_1(l, z)[0] + z * sp_hankel_1(l, z)[1]
    return normal, derivada

def mie_coefs(l, n1, n2, z):
    """
    mie_coefs(l, n1, n2, z)
    Coeficientes de Mie a_l y b_l.

    Depende de riccati_bessel y riccati_hankel

    l: orden del coeficiente (entero)
    n1: índice de refracción DE LA ESFERA (real, probar con complejo)
    n2: índice de refracción DEL MEDIO (real, probar con complejo)
    z: k*n2*a = 2*pi*n2*a/lambda, parámetro de tamaño, a es el radio de la esfera (real)

    Devuelve un vector con los coeficientes a_l y b_l (complejos)
    """
    M=n1/n2
    b_z = riccati_bessel(l, z)
    b_Mz = riccati_bessel(l, M*z)
    h_z = riccati_hankel_1(l, z)
    
    al = (M * b_Mz[0] * b_z[1] - b_z[0] * b_Mz[1]) / \
        (M * b_Mz[0] * h_z[1] - h_z[0] * b_Mz[1])
    bl = (M * b_Mz[1] * b_z[0] - b_z[1] * b_Mz[0]) / \
        (M * b_Mz[1] * h_z[0] - h_z[1] * b_Mz[0])
    
    return al, bl

def matriz_stokes_mie(l, wl, r, theta, phi, px, py):
    A = B = C = D = 0
    k = 2*np.pi/wl
    kr= k*r
    
    l=1
    gep, gem, gmp, gmm = BSC_plana(l, px, py) # G_TE, G_TM Neves => gm, ge !!!
  
    m=1
    

    legendre = assoc_legendre_p(l, m, np.cos(theta), diff_n = 1)
    pi = legendre[0]/np.sin(theta)
    tau = -np.sin(theta)*legendre[1]

    clm = (
        np.exp(1j*kr)/kr * 
        (-1j)**(l+2)/np.sqrt(l*(l+1)) * 
        np.sqrt((2*l+1)/(4*np.pi)) * 
        np.sqrt(f(l-m)/f(l+m)) *
        np.exp(1j*m*phi)
           )
    
    A += -gep * clm * tau
    B +=  gmp * clm * pi*1j*m
    C += -gep * clm * pi*1j*m
    D += -gmp * clm * tau 
 

    m=-1

    legendre = assoc_legendre_p(l, m, np.cos(theta), diff_n = 1)
    pi = legendre[0]/np.sin(theta)
    tau = -np.sin(theta)*legendre[1]
    
    clm = (
        np.exp(1j*kr)/kr * 
        (-1j)**(l+2)/np.sqrt(l*(l+1)) * 
        np.sqrt((2*l+1)/(4*np.pi)) * 
        np.sqrt(f(l-m)/f(l+m)) *
        np.exp(1j*m*phi)
           )
    
    
    A += -gem * clm * tau
    B +=  gmm * clm * pi*1j*m
    C += -gem * clm * pi*1j*m
    D += -gmm * clm * tau 

    M = np.zeros((4,4))
    
    M[0][0] = np.abs(A)**2 + np.abs(C)**2
    M[0][1] = np.abs(B)**2 + np.abs(D)**2
    M[0][2] = 2*(np.real(A*np.conj(B)) + np.real(C*np.conj(D)))
    M[0][3] = -2*(np.imag(A*np.conj(B)) + np.imag(C*np.conj(D)))
    
    M[1][0] = np.abs(A)**2 - np.abs(C)**2
    M[1][1] = np.abs(B)**2 - np.abs(D)**2
    M[1][2] = 2*(np.real(A*np.conj(B)) - np.real(C*np.conj(D)))
    M[1][3] = -2*(np.imag(A*np.conj(B)) - np.imag(C*np.conj(D)))
    
    M[2][0] = -2*np.real(A*np.conj(C))
    M[2][1] = -2*np.real(B*np.conj(D))
    M[2][2] = -2*(np.real(A*np.conj(D)) + np.real(B*np.conj(C)))
    M[2][3] = 2*(np.imag(A*np.conj(D)) - np.imag(B*np.conj(C)))
    
    M[3][0] = 2*np.imag(A*np.conj(C))
    M[3][1] = 2*np.imag(B*np.conj(D))
    M[3][2] = 2*(np.imag(A*np.conj(D)) + np.imag(B*np.conj(C)))
    M[3][3] = 2*(np.real(A*np.conj(D)) - np.real(B*np.conj(C)))
    

    
    return M


#%%
data_n=np.loadtxt("/Users/millanperez/Documents/wip/cmaps jorge/materiales/Si-n-AS.txt", delimiter=",")
data_k=np.loadtxt("/Users/millanperez/Documents/wip/cmaps jorge/materiales/Si-k-AS.txt", delimiter=",")

wl_start = 0.45
wl_end = 0.80
nwl = 100


wl_int = np.linspace(wl_start, wl_end, nwl)
n1_int = np.interp(wl_int, data_n[:,0], data_n[:,1])+1j*np.interp(wl_int, data_k[:,0], data_k[:,1]) #Con pérdidas
#n1_int = np.interp(wl_int, data_n[:,0], data_n[:,1]) #Sin pérdidas



#%%
nrad= 150

MA = np.zeros((nrad, nwl))
MB = np.zeros((nrad, nwl))
MC = np.zeros((nrad, nwl))
MD = np.zeros((nrad, nwl))


MA_ex = np.zeros((nrad, nwl))
MB_ex = np.zeros((nrad, nwl))
MC_ex = np.zeros((nrad, nwl))
MD_ex = np.zeros((nrad, nwl))




px = 1/np.sqrt(2)
py = 1j/np.sqrt(2)

n2 = 1

radv = np.linspace(20e-9, 80e-9, nrad)
z_imp = 376.86             # impedancia del medio
E0 = 1.0                # amplitud del campo
l_max = 3              # orden máximo del desarrollo

theta_grados = 70
phi_grados = 10

theta = theta_grados*2*np.pi/360
phi = phi_grados*2*np.pi/360
r = 10


for i, rad in enumerate(radv):
    if i % 10 == 0: print(f"Procesando i={i}")
    for j, wl in enumerate(wl_int):
        n1 = n1_int[j]
        wl = wl*1e-6
        k0 = 2*np.pi/wl          # número de onda
        k=n2*k0
        
        Er     = 0j
        Etheta = 0j
        Ephi   = 0j
        
        
        for l in range(1, l_max+1): 
            #    if l % 10 == 0: print(f"Procesando orden l={l}")
            G_TM_p, G_TM_m, G_TE_p, G_TE_m = BSC_plana(l, px, py)
            al, bl = mie_coefs(l, n1, n2, k*rad)
            al_p = -G_TM_p * al
            al_m = -G_TM_m * al
            bl_p = -G_TE_p * bl
            bl_m = -G_TE_m * bl
            
            Nr_p, Nt_p, Np_p = N_lm(l, 1, k, r, theta, phi, 'h')
            Nr_m, Nt_m, Np_m = N_lm(l, -1, k, r, theta, phi, 'h')
        
            Mr_p, Mt_p, Mp_p = M_lm(l,  1, k, r, theta, phi, 'h')
            Mr_m, Mt_m, Mp_m = M_lm(l, -1, k, r, theta, phi, 'h')
            
            Er     += al_p*Nr_p + al_m*Nr_m
            Etheta += al_p*Nt_p + al_m*Nt_m + bl_p*Mt_p + bl_m*Mt_m
            Ephi   += al_p*Np_p + al_m*Np_m + bl_p*Mp_p + bl_m*Mp_m
            
    
        s0 = np.abs(Etheta)**2 + np.abs(Ephi)**2
        s1 = np.abs(Etheta)**2 - np.abs(Ephi)**2
        s2 = -2*np.real(Etheta*np.conj(Ephi))
        s3 = 2*np.imag(Etheta*np.conj(Ephi))
        
        s = [s0, s1, s2, s3]
        
        M = matriz_stokes_mie(1, wl, r, theta, phi, px, py)
        
        M_inv = np.linalg.inv(M)
        
        Ms = M_inv.dot(s)
        
        MA[i][j]=Ms[0]
        MB[i][j]=Ms[1]
        MC[i][j]=Ms[2]
        MD[i][j]=Ms[3]
        
        a1, b1 = mie_coefs(1, n1, n2, k*rad)
       
        MA_ex[i][j] = np.abs(a1)**2 
        MB_ex[i][j] = np.abs(b1)**2
        MC_ex[i][j] = np.real(a1*np.conj(b1))
        MD_ex[i][j] = np.imag(a1*np.conj(b1))
      
        

#%% colormaps

RAD, WL = np.meshgrid(radv*1e9, wl_int)

title = "Calculados con expresión analítica - Silicio"

fig, axs = plt.subplots(2, 2, num=title, figsize=(16,9))
fig.suptitle(title)

plot00=axs[0, 0].pcolormesh(RAD, WL, MA_ex.transpose(), cmap='jet')
plt.colorbar(plot00, ax=axs[0,0])
axs[0, 0].set_title(r'$|a|^2$')
plot01=axs[0, 1].pcolormesh(RAD, WL, MB_ex.transpose(), cmap='jet')
plt.colorbar(plot01, ax=axs[0,1])
axs[0, 1].set_title(r'$|b|^2$')
plot10=axs[1, 0].pcolormesh(RAD, WL, MC_ex.transpose(), cmap='jet')
plt.colorbar(plot10, ax=axs[1,0])
axs[1, 0].set_title(r'Re(ab*)')
plot11=axs[1, 1].pcolormesh(RAD, WL, MD_ex.transpose(), cmap='jet')
plt.colorbar(plot11, ax=axs[1,1])
axs[1, 1].set_title(r'Im(ab*)')

for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

title = "Calculados a través de la matriz - Silicio sin pérdidas"

fig, axs = plt.subplots(2, 2, num=title, figsize=(16,9))
fig.suptitle(title)

plot00=axs[0, 0].pcolormesh(RAD, WL, MA.transpose(), cmap='jet', clim=(0,1))
plt.colorbar(plot00, ax=axs[0,0])
axs[0, 0].set_title(r'$|a|^2$')
plot01=axs[0, 1].pcolormesh(RAD, WL, MB.transpose(), cmap='jet', clim=(0,1))
plt.colorbar(plot00, ax=axs[0,1])
axs[0, 1].set_title(r'$|b|^2$')
plot10=axs[1, 0].pcolormesh(RAD, WL, MC.transpose(), cmap='jet')
plt.colorbar(plot00, ax=axs[1,0])
axs[1, 0].set_title(r'Re(ab*)')
plot11=axs[1, 1].pcolormesh(RAD, WL, MD.transpose(), cmap='jet')
plt.colorbar(plot11, ax=axs[1,1])
axs[1, 1].set_title(r'Im(ab*)')

for ax in axs.flat:
    ax.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

title = "Errores - Silicio sin pérdidas"

fig, axs = plt.subplots(2, 2, num=title, figsize=(16,9))
fig.suptitle(title)

plot00=axs[0, 0].pcolormesh(RAD, WL, np.abs(MA_ex.transpose()-MA.transpose()), cmap='jet')
plt.colorbar(plot00, ax=axs[0,0])
axs[0, 0].set_title(r'$|a|^2$')
plot01=axs[0, 1].pcolormesh(RAD, WL, np.abs(MB_ex.transpose()-MB.transpose()), cmap='jet')
plt.colorbar(plot00, ax=axs[0,1])
axs[0, 1].set_title(r'$|b|^2$')
plot10=axs[1, 0].pcolormesh(RAD, WL, np.abs(MC_ex.transpose()-MC.transpose()), cmap='jet')
plt.colorbar(plot00, ax=axs[1,0])
axs[1, 0].set_title(r'Re(ab*)')
plot11=axs[1, 1].pcolormesh(RAD, WL, np.abs(MD_ex.transpose()-MD.transpose()), cmap='jet')
plt.colorbar(plot11, ax=axs[1,1])
axs[1, 1].set_title(r'Im(ab*)')

for ax in axs.flat:
    ax.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

#%%

RAD, WL = np.meshgrid(radv*1e9, wl_int)
fs=(12,9)
title = f"$|a_1|^2$, $\\theta$ = {theta_grados}, n$_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
fig.suptitle(title)

plot00=axs[0].pcolormesh(RAD, WL, MA_ex.transpose(), cmap='jet', clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Expresión analítica')
plot01=axs[1].pcolormesh(RAD, WL, MA.transpose(), cmap='jet', clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Método matriz')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MA.transpose()-MA_ex.transpose())/MA.max()), cmap='inferno')
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Error relativo')


for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

title = f"$|b_1|^2$, $\\theta$ = {theta_grados}, n$_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
fig.suptitle(title)

plot00=axs[0].pcolormesh(RAD, WL, MB_ex.transpose(), cmap='jet', clim=(0,1))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Expresión analítica')
plot01=axs[1].pcolormesh(RAD, WL, MB.transpose(), cmap='jet', clim=(0,1))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Método matriz')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MB.transpose()-MB_ex.transpose())/MB.max()), cmap='inferno')
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Error relativo')


for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

title = f"$Re(a_1b_1*)$, $\\theta$ = {theta_grados}, n$_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
fig.suptitle(title)

plot00=axs[0].pcolormesh(RAD, WL, MC_ex.transpose(), cmap='jet', clim=(-0.2, 0.2))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Expresión analítica')
plot01=axs[1].pcolormesh(RAD, WL, MC.transpose(), cmap='jet', clim=(-0.2, 0.2))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Método matriz')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MC.transpose()-MC_ex.transpose())/MC.max()), cmap='inferno')
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Error relativo')


for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

title = f"$Im(a_1b_1*)$, $\\theta$ = {theta_grados}, n$_2$={n2}"

fig, axs = plt.subplots(3, num=title, figsize=fs)
fig.suptitle(title)

plot00=axs[0].pcolormesh(RAD, WL, MD_ex.transpose(), cmap='jet', clim=(-0.3, 0))
plt.colorbar(plot00, ax=axs[0])
axs[0].set_title(r'Expresión analítica')
plot01=axs[1].pcolormesh(RAD, WL, MD.transpose(), cmap='jet', clim=(-0.3, 0))
plt.colorbar(plot01, ax=axs[1])
axs[1].set_title(r'Método matriz')
plot10=axs[2].pcolormesh(RAD, WL, (np.abs(MD.transpose()-MD_ex.transpose())/MD.max()), cmap='inferno')
plt.colorbar(plot10, ax=axs[2])
axs[2].set_title(r'Error relativo')


for axi in axs.flat:
    axi.set(xlabel=r'Sphere radius ($n m$)', ylabel=r'Wavelength ($\mu m$)')

#%% Cortes en las gráficas

rads_i = [45e-9, 55e-9, 65e-9, 75e-9, 85e-9]
#rads_i = [75e-9]

#%%
plt.figure()
plt.title("Cálculo matriz")
for rad_i in rads_i:
    irad = np.argmin(np.abs(radv-rad_i))
    
    # print(radv[irad])
    
    MA_corte = MA[irad,:]
    MB_corte = MB[irad,:]
    
    Q_sca = (6*np.pi/(2*np.pi/wl_int)**2)*(MA_corte+MB_corte)
    
    plt.plot(wl_int, Q_sca, label=str(rad_i*2e9)+ " nm")
    
plt.xlabel("Wavelength")
plt.ylabel(r"Q$_{sca}$")
plt.legend()
plt.show()

plt.figure()
plt.title("Expresiones analíticas")
for rad_i in rads_i:
    irad = np.argmin(np.abs(radv-rad_i))
    
    # print(radv[irad])
    
    MA_corte = MA_ex[irad,:]
    MB_corte = MB_ex[irad,:]
    
    Q_sca = (6*np.pi/(2*np.pi/(n2*wl_int))**2)*(MA_corte+MB_corte)
    
    plt.plot(wl_int, Q_sca, label=str(rad_i*2e9)+" nm")

plt.xlabel("Wavelength")
plt.ylabel(r"Q$_{sca}$")
plt.legend()
plt.show()

#%%

plt.figure()
plt.title("Cálculo todos ordenes")


for rad_i in rads_i:
    irad = np.argmin(np.abs(radv-rad_i))
    
    # print(radv[irad])
    Q_sca=0
    k0 = 2*np.pi/(wl_int*1e-6)          # número de onda
    k=n2*k0
    for j in range(1,11):
        aj, bj = mie_coefs(j, n1_int, n2, k*rad_i)
        Q_sca += (2*np.pi/k**2)*(2*j+1)*(np.abs(aj)**2+np.abs(bj)**2)
#        plt.plot(wl_int, np.abs(aj)**2)
#        plt.plot(wl_int, np.abs(bj)**2)
    
    plt.plot(wl_int, Q_sca, label=str(rad_i*2e9)+ " nm")
    
plt.xlabel("Wavelength")
plt.ylabel(r"Q$_{sca}$")
plt.legend()
plt.show()