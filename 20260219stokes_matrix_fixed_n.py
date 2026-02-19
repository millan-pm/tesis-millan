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
wlv = np.linspace(1e-6, 2e-6, 300)                # longitud de onda
sc_a = []
sc_b = []
sc_t = []
al = []
bl = []


px = 1/np.sqrt(3)
py = 1j * np.sqrt(2/3)

n1 = 3.5
n2 = 1
rad = 230e-9
z_imp = 376.86             # impedancia del medio
E0 = 1.0                # amplitud del campo
l_max = 1              # orden máximo del desarrollo
nr = 1               # puntos en r
nphi = 1             # puntos en phi
ntheta = 1              # Solo 1 punto en theta

theta_grados = 144.5
phi_grados = 17.4

theta = theta_grados*2*np.pi/360
phi = phi_grados*2*np.pi/360
r = 10

for j,wl in enumerate(wlv):
    k = 2*np.pi/wl          # número de onda
    
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

    sc_a.append(2*np.pi/(2*np.pi/wl)**2 * 3 * Ms[0]/rad**2)
    sc_b.append(2*np.pi/(2*np.pi/wl)**2 * 3 * Ms[1]/rad**2)
    sc_t.append(2*np.pi/(2*np.pi/wl)**2 * 3 * (Ms[0]+Ms[1])/rad**2)

sc_a = np.array(sc_a)
sc_b = np.array(sc_b)
sc_t = np.array(sc_t)

plt.figure(figsize=(12,6))

#plt.plot(wlv*1e9, sc_t, label="$\\sigma_{a1}+\sigma_{b1}$")
plt.plot(wlv*1e9, sc_a, label="$\\sigma_{a1}$")
plt.plot(wlv*1e9, sc_b, label="$\\sigma_{b1}$")

plt.title(f"px = {px}, py = {py}, $\\theta$ = {theta_grados}$^\\circ$, l = {l_max}")

plt.xlabel("$\\lambda$ (nm)")
plt.ylabel("$\\sigma/a^2$")
plt.legend()
