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
    px = 1/np.sqrt(2)
    py = 1j/np.sqrt(2)

    G_TM_p = common * (-1j*px-py) 
    G_TM_m = common * (+1j*px-py) 
    G_TE_p = common * (px - 1j*py)
    G_TE_m = common * (px + 1j*py)
    
    return G_TM_p, G_TM_m, G_TE_p, G_TE_m

#%%

wl = 1.0                # longitud de onda
k = 2*np.pi/wl          # número de onda
z_imp = 1.0             # impedancia del medio
E0 = 1.0                # amplitud del campo
l_max = 70               # orden máximo del desarrollo
nr = 150               # puntos en r
nphi = 361             # puntos en phi
ntheta = 1              # Solo 1 punto en theta
r = np.linspace(1e-6, 10*wl, nr)
theta = np.linspace(np.pi/2, np.pi/2, ntheta) 
phi   = np.linspace(1e-6, 2*np.pi, nphi) 

R = r[:, None, None]         # forma (nr,1,1)
Theta = theta[None, :, None] # forma (1,1,1)
Phi = phi[None, None, :]     # forma (1,1,nphi)

#%%

Er     = np.zeros((nr, ntheta, nphi), dtype=np.complex64)
Etheta = np.zeros_like(Er)
Ephi   = np.zeros_like(Er)

Hr     = np.zeros((nr, ntheta, nphi), dtype=np.complex64)
Htheta = np.zeros_like(Hr)
Hphi   = np.zeros_like(Hr)

for l in range(1, l_max+1): 
    if l % 10 == 0: print(f"Procesando orden l={l}")
#    print(f"Procesando orden l={l}")
    
    G_TM_p, G_TM_m, G_TE_p, G_TE_m = BSC_plana(l)

    Nr_p, Nt_p, Np_p = N_lm(l, 1, k, R, Theta, Phi, 'j')
    Nr_m, Nt_m, Np_m = N_lm(l, -1, k, R, Theta, Phi, 'j')

    Mr_p, Mt_p, Mp_p = M_lm(l,  1, k, R, Theta, Phi, 'j')
    Mr_m, Mt_m, Mp_m = M_lm(l, -1, k, R, Theta, Phi, 'j')

    Er     += G_TM_p*Nr_p + G_TM_m*Nr_m
    Etheta += G_TM_p*Nt_p + G_TM_m*Nt_m + G_TE_p*Mt_p + G_TE_m*Mt_m
    Ephi   += G_TM_p*Np_p + G_TM_m*Np_m + G_TE_p*Mp_p + G_TE_m*Mp_m
    
    Hr     += (E0/z_imp)*(-G_TE_p*Nr_p - G_TE_m*Nr_m)
    Htheta += (E0/z_imp)*(-G_TE_p*Nt_p - G_TE_m*Nt_m + G_TM_p*Mt_p + G_TM_m*Mt_m)
    Hphi   += (E0/z_imp)*(-G_TE_p*Np_p - G_TE_m*Np_m + G_TM_p*Mp_p + G_TM_m*Mp_m)

#%%

Ex = (Er*np.sin(Theta)*np.cos(Phi) +
      Etheta*np.cos(Theta)*np.cos(Phi) -
      Ephi*np.sin(Phi))
    
Ey = (Er*np.sin(Theta)*np.sin(Phi) +
      Etheta*np.cos(Theta)*np.sin(Phi) +
      Ephi*np.cos(Phi))
    
Ez = Er*np.cos(Theta) - Etheta*np.sin(Theta)


Ex = np.squeeze(Ex)
Ey = np.squeeze(Ey)
Ez = np.squeeze(Ez)

E_plot = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
E_plot = np.squeeze(np.abs(Ephi))

Hx = (Hr*np.sin(Theta)*np.cos(Phi) +
      Htheta*np.cos(Theta)*np.cos(Phi) -
      Hphi*np.sin(Phi))
    
Hy = (Hr*np.sin(Theta)*np.sin(Phi) +
      Htheta*np.cos(Theta)*np.sin(Phi) +
      Hphi*np.cos(Phi))
    
Hz = Hr*np.cos(Theta) - Htheta*np.sin(Theta)


Hx = np.squeeze(Hx)
Hy = np.squeeze(Hy)
Hz = np.squeeze(Hz)

H_plot = np.sqrt(np.abs(Hx)**2 + np.abs(Hy)**2 + np.abs(Hz)**2)

#%%
r_i = wl
phi_i = 1e-6

ir = np.abs(np.argmin(r-r_i))
iphi = np.abs(np.argmin(phi-phi_i))


s0 = np.abs(Ex)**2 + np.abs(Ey)**2
s1 = np.abs(Ex)**2 - np.abs(Ey)**2
s2 = -2*np.real(Ex*np.conj(Ey))
s3 = 2*np.imag(Ex*np.conj(Ey))

S = [s0.max(), s1.max(), s2.max(), s3.max()]

print(S)

#%%

R_mesh, P_mesh = np.meshgrid(r, phi, indexing='ij')

X = R_mesh * np.cos(P_mesh)
Y = R_mesh * np.sin(P_mesh)

plt.figure(figsize=(10, 8))
plt.pcolormesh(X, Y, E_plot, shading='auto', cmap='jet', vmax = 2, vmin = 0)
plt.colorbar(label='abs(E)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'abs(E) en Plano XY (l_max={l_max})')
plt.axis('equal')
plt.show()

plt.figure(figsize=(10, 8))
plt.pcolormesh(X, Y, H_plot, shading='auto', cmap='jet', vmax = 2, vmin = 0)
plt.colorbar(label='abs(H)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'abs(H) en Plano XY (l_max={l_max})')
plt.axis('equal')
plt.show()

