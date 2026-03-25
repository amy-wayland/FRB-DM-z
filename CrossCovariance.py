import numpy as np
import pyccl as ccl
import HaloProfiles as hp

from pyccl.halos import Profile2pt
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.special import spherical_jn, eval_legendre

#%%
# Fiducial Cosmology

#%%

COSMO_P18 = {"Omega_c": 0.26066676,
             "Omega_b": 0.048974682,
             "h": 0.6766,
             "n_s": 0.9665,
             "sigma8": 0.8102,
             "matter_power_spectrum": "halofit"}
             
cosmo = ccl.Cosmology(**COSMO_P18)
cosmo.compute_growth()

#%%
# Halo Model Implementation

#%%

prof2pt = Profile2pt()
hmd_200c = ccl.halos.MassDef200c
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200c)
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200c)
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200c)
pM_nfw = ccl.halos.HaloProfileNFW(mass_def=hmd_200c, concentration=cM)
pM_bar = hp.HaloProfileNFWBaryon(mass_def=hmd_200c, concentration=cM, lMc=14.0, beta=0.6, A_star=0.03, eta_b=0.5)
pE = hp.HaloProfileDensityHE(mass_def=hmd_200c, concentration=cM, lMc=14.0, beta=0.6, A_star=0.03, eta_b=0.5)
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200c, log10M_max=15.0, log10M_min=10.0, nM=32)

#%%
# Functions To Compute Bispectrum

#%%

def rho_e_bar(a):
    return pE.get_normalization(cosmo, a, hmc=hmc)

def I_1_1(k, a):
    return hmc.I_1_1(cosmo, k, a, prof=pE) / rho_e_bar(a)

def I_1_2(k1, k2, a):
    k = np.array([k1, k2])
    return hmc.I_1_2(cosmo, k, a, prof=pE, prof_2pt=prof2pt, diag=False)[0, 1] / rho_e_bar(a)**2

def I_0_3(k1, k2, k3, a):
    
    norm = rho_e_bar(a)
    
    def integrand(M):
        u1 = pE.fourier(cosmo, k1, M, a) / norm
        u2 = pE.fourier(cosmo, k2, M, a) / norm
        u3 = pE.fourier(cosmo, k3, M, a) / norm
        return u1 * u2 * u3
    
    return hmc.integrate_over_massfunc(integrand, cosmo, a)

def P_lin(k, a):
    return ccl.linear_matter_power(cosmo, k, a)

def F2(k1, k2, mu):
    return (5/7 + 0.5 * (k1/k2 + k2/k1) * mu + 2/7 * mu**2)

def B_tree(k1, k2, k3, a):

    mu12 = (k3**2 - k1**2 - k2**2) / (2 * k1 * k2)
    mu23 = (k1**2 - k2**2 - k3**2) / (2 * k2 * k3)
    mu31 = (k2**2 - k3**2 - k1**2) / (2 * k3 * k1)
    
    P1 = P_lin(k1, a)
    P2 = P_lin(k2, a)
    P3 = P_lin(k3, a)
    
    return (2 * F2(k1, k2, mu12) * P1 * P2 +
            2 * F2(k2, k3, mu23) * P2 * P3 +
            2 * F2(k3, k1, mu31) * P3 * P1)

def B_1h(k1, k2, k3, a):
    return I_0_3(k1, k2, k3, a)

def B_2h(k1, k2, k3, a):
    return (I_1_1(k1, a) * I_1_2(k2, k3, a) * P_lin(k1, a) +
            I_1_1(k2, a) * I_1_2(k3, k1, a) * P_lin(k2, a) +
            I_1_1(k3, a) * I_1_2(k1, k2, a) * P_lin(k3, a))
    
def B_3h(k1, k2, k3, a):
    B_m = B_tree(k1, k2, k3, a)
    return I_1_1(k1, a) * I_1_1(k2, a) * I_1_1(k3, a) * B_m

def B_e(k1, k2, k3, a):
    return B_1h(k1, k2, k3, a) + B_2h(k1, k2, k3, a) + B_3h(k1, k2, k3, a)

#%%
# FRB Kernel

#%%

# FRB redshift distribution
alpha = 3.5
zz = np.linspace(0, 2, 128)
aa = 1/(1+zz)
nz = zz**2 * np.exp(-alpha*zz)
nz = nz/np.trapezoid(nz, zz)

# Want [G] = [cm^3 kg^{-1} s^{-2}]
G_m3_per_kg_per_s2 = ccl.physical_constants.GNEWT
G_cm3_per_kg_per_s2 = 1e6 * G_m3_per_kg_per_s2
G = G_cm3_per_kg_per_s2

# Want [m_p] = [kg]
mp_kg = 1.67262e-27
mp = mp_kg

# Want [H_0] = [Mpc s^{-1} Mpc^{-1}]
pc = 3.0857e13 # 1pc in km
km_to_Mpc = 1/(1e6*pc) # 1 km = 3.24078e-20 Mpc
H0_per_s = cosmo['H0'] * km_to_Mpc
H0 = H0_per_s

# Prefactor in units of [A] = [cm^{-3}]
xH = 0.75
A = (3*cosmo['Omega_b']*H0**2)/(8*np.pi*G*mp) * (1+xH)/2

# Cumulative integral of n(z)
nz_integrated = 1 - cumulative_trapezoid(nz, zz, initial=0)

# [W_{\chi}] = [A] = [cm^{-3}]
# Factor of 1e6 so that Cl is in units of [pc cm^{-3}]
h = cosmo['H0'] / 100
chis = ccl.comoving_radial_distance(cosmo, aa)
W_chi = A * (1+zz) * nz_integrated * 1e6

#%%
# Interpolate Kernel

#%%

chi_of_z_interp = interp1d(zz, chis, bounds_error=False, fill_value="extrapolate")
z_of_chi_interp = interp1d(chis, zz, bounds_error=False, fill_value="extrapolate")
W_interp = interp1d(chis, W_chi, bounds_error=False, fill_value=0.0)

def E_of_chi(chi):
    z = z_of_chi_interp(chi)
    a = 1/(1+z)
    return ccl.h_over_h0(cosmo, a)
    
#%%
# Evaluate 4D Integral

#%%

def covariance_DM_Cl(ell, z1, z2, z3, Nchi=20, Nmu=50):
    '''
    Compute the cross-covariance between 
    the DM-DM angular power spectrum
    and the DM-redshift relation.

    '''
    chi1_max = ccl.comoving_radial_distance(cosmo, 1/(1+z1))
    chi2_max = ccl.comoving_radial_distance(cosmo, 1/(1+z2))
    chi3_max = ccl.comoving_radial_distance(cosmo, 1/(1+z3))
    chi_H = ccl.physical_constants.CLIGHT_HMPC
    
    # Grids
    chi_min = 1e-2
    chi1_arr = np.linspace(chi_min, chi1_max, Nchi)
    chi2_arr = np.linspace(chi_min, chi2_max, Nchi)
    chi3_arr = np.linspace(chi_min, chi3_max, Nchi)
    mu_arr   = np.linspace(-1, 1, Nmu)
    
    dchi1 = chi1_arr[1] - chi1_arr[0]
    dchi2 = chi2_arr[1] - chi2_arr[0]
    dchi3 = chi3_arr[1] - chi3_arr[0]
    
    result = 0.0
    prefactor = ((ell + 0.5)**3) / (4 * np.pi**2) * chi_H**3
    
    for chi1 in chi1_arr:
        z1p = z_of_chi_interp(chi1)
        a1 = 1 / (1 + z1p)
        W1 = W_interp(chi1)
        E1 = E_of_chi(chi1)
        
        for chi2 in chi2_arr:
            W2 = W_interp(chi2)
            E2 = E_of_chi(chi2)
            k2 = (ell + 0.5) / chi2
            
            for chi3 in chi3_arr:
                W3 = W_interp(chi3)
                E3 = E_of_chi(chi3)
                k3 = (ell + 0.5) / chi3
                k1_sq = np.maximum(k2**2 + k3**2 - 2*k2*k3*mu_arr, 1e-6)
                k1 = np.sqrt(k1_sq)
                
                # Spherical Bessel
                j0_vals = spherical_jn(0, k1 * chi1)
                
                # Legendre
                P_ell = eval_legendre(ell, mu_arr)
                
                # Bispectrum
                k1 = np.clip(k1, 1e-3, 1e2)
                k2 = np.clip(k2, 1e-3, 1e2)
                k3 = np.clip(k3, 1e-3, 1e2)
                B_vals = np.array([B_e(k1[i], k2, k3, a1) for i in range(len(mu_arr))])
                
                # Evaluate mu integral
                mu_integral = np.trapezoid(j0_vals * P_ell * B_vals, mu_arr)
                weight = (W1/E1 * W2/(chi2 * E2) * W3/(chi3**2 * E3))
                result += weight * mu_integral
    
    result *= prefactor * dchi1 * dchi2 * dchi3
    return result
