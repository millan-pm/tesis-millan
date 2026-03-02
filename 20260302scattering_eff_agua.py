"""
Created on Mon Feb  2 11:36:17 2026

@author: milla

Cálculo de eficiencia de Scattering y parámetro de asimetría.

Multipolos precalculados para agilizar la ejecución.
"""

import numpy as np
from scipy.special import sph_harm, spherical_jn, spherical_yn, sph_harm_y
import matplotlib.pyplot as plt

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
    
    nr = R.shape[0]
    ntheta = Theta.shape[1]
    nphi = Phi.shape[2]

    M_r     = np.zeros((nr, ntheta, nphi), dtype=np.complex64)
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

def BSC_plana(l):
    common = np.sqrt(np.pi*(2*l+1)) * 1j**(l) 
    px = 0
    py = 1
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
#%% Inicialización de variables
a_points = 1
wl_points = 200

av = np.linspace(80e-9, 80e-9, a_points)                 # size parameter
n1 = 5                   # índice de la esfera
n2 = 1.33                 # índice del medio    
wlv = np.linspace(400e-9, 800e-9, wl_points)               # longitud de onda          # número de onda

z_imp = n2*376.86          # impedancia del medio
E0 = 1.0                # amplitud del campo
l_max = 6            # orden máximo del desarrollo
nr = 1                  # puntos en r
nphi = 361              # puntos en phi
ntheta = 181            # puntos en theta

r = np.linspace(10, 10, nr)
theta = np.linspace(1e-6, np.pi-1e-6, ntheta) 
phi   = np.linspace(1e-6, 2*np.pi-1e-6, nphi) 

dt = theta[1]-theta[0]
dp = phi[1]-phi[0]

R = r[:, None, None]            # forma (nr,1,1)
Theta = theta[None, :, None]    # forma (1,1,1)
Phi = phi[None, None, :]        # forma (1,1,nphi)

#%% Cálculo de los campos y eficiencia de scattering
C_sca_poynting=np.zeros((a_points, wl_points))
Q_sca_poynting=np.zeros((a_points, wl_points))
g = np.zeros((a_points, wl_points))
for i, a in enumerate(av):
    for j, wl in enumerate(wlv):
        if j % 10 == 0: print(f"Procesando j={j}")
        k0=2*np.pi/wl
        k=n2*k0
        q = k*a
        
        Er     = np.zeros((nr, ntheta, nphi), dtype=np.complex64)
        Etheta = np.zeros_like(Er)
        Ephi   = np.zeros_like(Er)
        
        Hr     = np.zeros((nr, ntheta, nphi), dtype=np.complex64)
        Htheta = np.zeros_like(Hr)
        Hphi   = np.zeros_like(Hr)
        
        for l in range(1, l_max+1): 
            #    print(f"Procesando orden l={l}")
            G_TM_p, G_TM_m, G_TE_p, G_TE_m = BSC_plana(l)
            al, bl = mie_coefs(l, n1, n2, q)
            al_p = -G_TM_p * al
            al_m = -G_TM_m * al
            bl_p = -G_TE_p * bl
            bl_m = -G_TE_m * bl
            
            Nr_p, Nt_p, Np_p = N_lm(l,  1, k, R, Theta, Phi, 'h')
            Nr_m, Nt_m, Np_m = N_lm(l, -1, k, R, Theta, Phi, 'h')
            Mr_p, Mt_p, Mp_p = M_lm(l,  1, k, R, Theta, Phi, 'h')
            Mr_m, Mt_m, Mp_m = M_lm(l, -1, k, R, Theta, Phi, 'h')
            
            
            Er     += al_p*Nr_p + al_m*Nr_m
            Etheta += al_p*Nt_p + al_m*Nt_m + bl_p*Mt_p + bl_m*Mt_m
            Ephi   += al_p*Np_p + al_m*Np_m + bl_p*Mp_p + bl_m*Mp_m
            
            Hr     += (E0/z_imp)*(-bl_p*Nr_p - bl_m*Nr_m)
            Htheta += (E0/z_imp)*(-bl_p*Nt_p - bl_m*Nt_m + al_p*Mt_p + al_m*Mt_m)
            Hphi   += (E0/z_imp)*(-bl_p*Np_p - bl_m*Np_m + al_p*Mp_p + al_m*Mp_m)
        
        Sr = np.squeeze( 0.5 * np.real(Etheta * np.conj(Hphi) - Ephi * np.conj(Htheta)) )
        
        integrando = Sr * R**2 * np.sin(Theta)
        P_sca = np.sum(integrando * dt * dp)
        P_sca_weighted = np.sum(integrando *np.cos(Theta)* dt * dp)
        
        I_inc = np.abs(E0)**2 / (2 * z_imp)
        
        C_sca_poynting[i][j] = P_sca / I_inc
        g[i][j] = P_sca_weighted / P_sca
        a = q / k
        Q_sca_poynting[i][j] = C_sca_poynting[i][j] / (np.pi * a**2)


#%%

Q_sca=0
k0 = 2*np.pi/(wlv)          # número de onda
k=n2*k0

for j in range(1,l_max+1):
    aj, bj = mie_coefs(j, n1, n2, k*a)
    Q_sca += (2*np.pi/k**2)*(2*j+1)*(np.abs(aj)**2+np.abs(bj)**2)


plt.figure(figsize=(8,6))    
plt.plot(wlv*1e9, Q_sca/(np.pi*a**2))
plt.title(f"analítico, n2={n2}")
plt.xlabel("wavelength (nm)")
plt.ylabel("Q_sca")
plt.show()

plt.figure(figsize=(8,6))
plt.plot(wlv*1e9, Q_sca_poynting[0,:])
plt.title(f"poynting, n2={n2}")
plt.xlabel("wavelength (nm)")
plt.ylabel("Q_sca")
plt.show


#%% 

g_mie=0
k0 = 2*np.pi/(wlv)          # número de onda
k=n2*k0

for l in range(1, l_max):
    al, bl = mie_coefs(l, n1, n2, k*a)
    al1, bl1 = mie_coefs(l+1, n1, n2, k*a)
    
    g_mie += (
    ((l*(l+2))/(l+1))*np.real(al*np.conj(al1)+bl*np.conj(bl1))
    + ((2*l+1)/(l*(l+1)))*np.real(al*np.conj(bl))
    )
    
g_mie = (4/(k*a)**2)*g_mie*(np.pi*a**2)/Q_sca

plt.figure(figsize=(8,6))    
plt.plot(wlv*1e9, g_mie)
plt.title(f"analítico, n2={n2}")
plt.xlabel("wavelength (nm)")
plt.ylabel("g")
plt.show()

plt.figure(figsize=(8,6))
plt.plot(wlv*1e9, g[0,:])
plt.title(f"poynting, n2={n2}")
plt.xlabel("wavelength (nm)")
plt.ylabel("g")
plt.show