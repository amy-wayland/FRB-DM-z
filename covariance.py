import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
from scipy.special import spherical_jn, eval_legendre, jv
from scipy.integrate import cumulative_trapezoid, simpson
from core import cosmo, P_e, load_bispectrum, P_e_array

# -----------------------------------------------------------
# Load Bispectrum
# -----------------------------------------------------------

B_interp, data = load_bispectrum("bispectrum_grid.npz")
k_grid = data["k_grid"]

# -----------------------------------------------------------
# FRB Kernel
# -----------------------------------------------------------

# FRB redshift distribution
alpha = 3.5
zz = np.linspace(0, 2, 128)
aa = 1/(1+zz)
nz = zz**2 * np.exp(-alpha*zz)
nz = nz/np.trapz(nz, zz)

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

# Interpolate Kernel
chi_of_z_interp = interp1d(zz, chis, bounds_error=False, fill_value="extrapolate")
z_of_chi_interp = interp1d(chis, zz, bounds_error=False, fill_value="extrapolate")
W_interp = interp1d(chis, W_chi, bounds_error=False, fill_value=0.0)


k_interp = np.geomspace(1e-4,1e2,500)
a_interp = np.linspace(.2, 1, 100)
Pe_arr = P_e_array(k_interp, a_interp)
Pe_interpolator = ccl.pk2d.Pk2D(a_arr = a_interp,
                                lk_arr = np.log(k_interp),
                                pk_arr = np.log(Pe_arr),
                                extrap_order_lok=1,
                                extrap_order_hik=1,
                                is_logp=True)



def E_of_chi(chi):
    z = z_of_chi_interp(chi)
    a = 1/(1+z)
    return ccl.h_over_h0(cosmo, a)

# -----------------------------------------------------------
# DM-z Auto-Covariance
# -----------------------------------------------------------

def W_single_FRB(chi, chi_s):
    '''
    DM kernel for a single FRB at comoving distance chi_s.
    W_D(chi) = A * (1+z(chi)) for chi < chi_s, else 0.
    '''
    if chi >= chi_s:
        return 0.0
    z = float(z_of_chi_interp(chi))
    return A * (1+z) * 1e6  # same units as W_chi [pc cm^{-3}]

import time

def C_ij_ell(ell, zi, zj, Nchi=100):
    '''
    Angular power spectrum C_ij(ell) under the Limber approximation.
    '''
    chi_i = float(ccl.comoving_radial_distance(cosmo, 1/(1+zi)))
    chi_j = float(ccl.comoving_radial_distance(cosmo, 1/(1+zj)))
    chi_max = min(chi_i, chi_j)
    chi_arr = np.linspace(1e-2, chi_max, Nchi)

    z_arr = z_of_chi_interp(chi_arr)
    a_arr = 1 / (1 + z_arr)
    k_arr = np.clip((ell + 0.5) / chi_arr, 1e-3, 1e2)
    # All chi in chi_arr <= chi_max <= min(chi_i, chi_j), so both kernels are non-zero
    W_arr = A * (1 + z_arr) * 1e6
    Pe_arr = np.diag(Pe_interpolator(k_arr,a_arr))
    integrand = W_arr**2 * Pe_arr / chi_arr**2
    return np.trapz(integrand, chi_arr)

def cov_DD(zi, zj, cos_theta, ell_max=500, Nchi=100, flat_sky=True):
    '''
    DM-DM auto-covariance summed over multipoles.

    By default this uses the original full-sky Legendre expansion.
    If flat_sky=True, it uses the flat-sky Bessel-integral approximation:
        Cov(theta) = int d ell [ell / (2 pi)] J_0(ell theta) C_ij(ell).
    '''
    ell_arr = np.unique(np.round(np.logspace(0, np.log10(500), 100)).astype(int))
    C_ell_arr = np.array([C_ij_ell(ell, zi, zj, Nchi=Nchi) for ell in ell_arr])

    if not flat_sky:
        P_ell_arr = np.array([float(eval_legendre(ell, cos_theta)) for ell in ell_arr])
        integrand = (2 * ell_arr + 1) / (4 * np.pi) * P_ell_arr * C_ell_arr
        return np.trapz(integrand, ell_arr)

    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    ell_integral = np.geomspace(ell_arr[0], ell_arr[-1], int(1e5))
    C_ell_interp = np.interp(ell_integral, ell_arr, C_ell_arr)
    bessel = jv(0, ell_integral * theta)
    integrand = ell_integral * C_ell_interp * bessel

    return simpson(integrand, x=ell_integral) / (2.0 * np.pi)

# -----------------------------------------------------------
# Cl^{DD} Auto-Covariance
# -----------------------------------------------------------

def C_ell_DD(ell, Nchi=100):
    '''
    DM angular power spectrum C_ell^DD:
    C_ell^DD = int dchi/chi^2 * W(chi)^2 * P_e((ell+0.5)/chi, z(chi))
    '''
    chi_max = float(ccl.comoving_radial_distance(cosmo, 1/(1+2.0)))
    chi_arr = np.linspace(1e-2, chi_max, Nchi)

    z_arr = z_of_chi_interp(chi_arr)
    a_arr = 1 / (1 + z_arr)
    k_arr = np.clip((ell + 0.5) / chi_arr, 1e-3, 1e2)
    W_arr = W_interp(chi_arr)
    Pe_arr = np.diag(Pe_interpolator(k_arr,a_arr))
    integrand = W_arr**2 * Pe_arr / chi_arr**2
    return np.trapz(integrand, chi_arr)

def cov_ClCl(ell, ell_prime, f_sky=1.0, Nchi=100, delta_ell=1):
    '''
    Gaussian (Knox) covariance of the DM power spectrum.
    '''
    if ell != ell_prime:
        return 0.0
    C = C_ell_DD(ell, Nchi=Nchi)
    return 2.0 / (2*ell + 1) / f_sky  * C**2

# -----------------------------------------------------------
# Cross-Covariance
# -----------------------------------------------------------

Mpc_to_pc = 3.0857e18

def covariance_DM_Cl(ell, z1, z2, z3, Nchi=20, Nmu=40):
    '''
    Compute the cross-covariance between the DM-DM angular power spectrum
    and the DM-redshift relation.
    '''
    chi1_max = ccl.comoving_radial_distance(cosmo, 1/(1+z1))
    chi2_max = ccl.comoving_radial_distance(cosmo, 1/(1+z2))
    chi3_max = ccl.comoving_radial_distance(cosmo, 1/(1+z3))
    chi_H = ccl.physical_constants.CLIGHT_HMPC

    # Set chi_min so that k = (ell+0.5)/chi stays within the bispectrum grid
    # k_grid max = 10 Mpc^{-1}, so chi_min = (ell+0.5)/10
    k_grid_max = float(np.max(k_grid))
    k_grid_min = float(np.min(k_grid))
    chi_min_ell = (ell + 0.5) / k_grid_max
    chi_max_ell = (ell + 0.5) / k_grid_min

    # Clip chi ranges to keep k within bispectrum grid
    chi1_arr = np.linspace(max(chi_min_ell, 1e-2), min(chi1_max, chi_max_ell), Nchi)
    chi2_arr = np.linspace(max(chi_min_ell, 1e-2), min(chi2_max, chi_max_ell), Nchi)
    chi3_arr = np.linspace(max(chi_min_ell, 1e-2), min(chi3_max, chi_max_ell), Nchi)
    mu_arr = np.linspace(-1, 1, Nmu)
    Pell_arr = eval_legendre(ell, mu_arr)

    # Check grids are valid
    if chi1_arr[0] >= chi1_arr[-1] or len(chi1_arr) < 2:
        return 0.0
    if chi2_arr[0] >= chi2_arr[-1] or len(chi2_arr) < 2:
        return 0.0
    if chi3_arr[0] >= chi3_arr[-1] or len(chi3_arr) < 2:
        return 0.0

    # Precompute geometry
    chi_s = chi1_max
    z1_arr = z_of_chi_interp(chi1_arr)
    a1_arr = 1 / (1 + z1_arr)
    W1_arr = np.where(chi1_arr < chi_s, A * (1 + z1_arr) * 1e6, 0.0)
    W2_arr = W_interp(chi2_arr)
    W3_arr = W_interp(chi3_arr)

    E1_arr = E_of_chi(chi1_arr)
    E2_arr = E_of_chi(chi2_arr)
    E3_arr = E_of_chi(chi3_arr)

    k2_arr = (ell + 0.5) / chi2_arr
    k3_arr = (ell + 0.5) / chi3_arr

    dchi1 = chi1_arr[1] - chi1_arr[0]
    dchi2 = chi2_arr[1] - chi2_arr[0]
    dchi3 = chi3_arr[1] - chi3_arr[0]

    G1 = W1_arr / E1_arr
    G2 = W2_arr / (chi2_arr * E2_arr)
    G3 = W3_arr / (chi3_arr**2 * E3_arr)

    phi_arr = np.arccos(mu_arr)  # shape: (Nmu,)
    prefactor = ((ell + 0.5)**3) / (4 * np.pi**2) * chi_H**3

    # Build all (i1, i2, i3, mu) evaluation points for B_interp at once.
    # B_interp axes: (a, k2, k3, phi). Broadcast to shape (Nchi, Nchi, Nchi, Nmu).
    a1_pts  = np.broadcast_to(a1_arr[:, None, None, None], (Nchi, Nchi, Nchi, Nmu))
    k2_pts  = np.broadcast_to(k2_arr[None, :, None, None], (Nchi, Nchi, Nchi, Nmu))
    k3_pts  = np.broadcast_to(k3_arr[None, None, :, None], (Nchi, Nchi, Nchi, Nmu))
    phi_pts = np.broadcast_to(phi_arr[None, None, None, :], (Nchi, Nchi, Nchi, Nmu))

    pts = np.stack([a1_pts.ravel(), k2_pts.ravel(),
                    k3_pts.ravel(), phi_pts.ravel()], axis=-1)
    B_4d = B_interp(pts).reshape(Nchi, Nchi, Nchi, Nmu)

    # k1 and j0: shape (Nchi, Nchi, Nchi, Nmu)
    k2_bc   = k2_arr[None, :, None, None]
    k3_bc   = k3_arr[None, None, :, None]
    mu_bc   = mu_arr[None, None, None, :]
    chi1_bc = chi1_arr[:, None, None, None]

    k1_4d = np.clip(
        np.sqrt(k2_bc**2 + k3_bc**2 + 2*k2_bc*k3_bc*mu_bc),
        k_grid_min, k_grid_max)
    j0_4d = spherical_jn(0, k1_4d * chi1_bc)

    # Integrate over mu: shape (Nchi, Nchi, Nchi)
    mu_integrals = np.trapz(
        j0_4d * Pell_arr[None, None, None, :] * B_4d, mu_arr, axis=-1)

    G_3d = G1[:, None, None] * G2[None, :, None] * G3[None, None, :]
    result = np.sum(G_3d * mu_integrals) * prefactor * dchi1 * dchi2 * dchi3
    return result / Mpc_to_pc

# -----------------------------------------------------------
# Build Full Covariance
# -----------------------------------------------------------

def build_covariance_matrix(ell, z_frb, cos_theta_matrix,
                            f_sky=0.7, Nchi=50, Nmu=40,
                            flat_sky=False, delta_ell=1):
    '''
    Build the full (N+1)x(N+1) covariance matrix:
    C = [Cov[D_i, D_j]    Cov[D_i, C_ell]  ]
        [Cov[C_ell, D_j]  Cov[C_ell, C_ell]].
    '''
    N = len(z_frb)
    Ntot = N + 1
    cov = np.zeros((Ntot, Ntot))

    # Block 1: Cov[D_i, D_j], shape (N, N)
    print("Computing Cov[D_i, D_j]...")
    for i in range(N):
        for j in range(i, N):
            val = cov_DD(
                z_frb[i], z_frb[j], cos_theta_matrix[i, j],
                Nchi=Nchi, flat_sky=flat_sky
            )
            cov[i, j] = val
            cov[j, i] = val
            print(f"  ({i},{j}): {val:.3e}")#, end='\r')
    print()

    # Blocks 2 and 3: Cov[D_i, C_ell],  shape (N, 1) and (1, N)
    # z2=z3=z_max: upper limits of chi2, chi3 integrals correspond to
    # the maximum redshift of the survey over which C_ell^DD is defined.
    print("Computing Cov[D_i, C_ell]...")
    z_max = float(zz[-1])
    for i in range(N):
        val = covariance_DM_Cl(ell, z_frb[i], z_max, z_max,
                                Nchi=Nchi, Nmu=Nmu)
        cov[i, N] = val
        cov[N, i] = val
        print(f"  FRB {i} (z={z_frb[i]:.2f}): {val:.3e}", end='\r')
    print()

    # Block 4: Cov[C_ell, C_ell], scalar
    print("Computing Cov[C_ell, C_ell]...")
    cov[N, N] = cov_ClCl(ell, ell, f_sky=f_sky, Nchi=Nchi, delta_ell=delta_ell)
    print(f"  {cov[N,N]:.3e}")

    # Correlation coefficient
    diag = np.sqrt(np.diag(cov))
    corr = cov / np.outer(diag, diag)

    return cov, corr

# -----------------------------------------------------------
# Plot Correlation Matrix
# -----------------------------------------------------------

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "axes.linewidth": 1.2
})

def plot_correlation_matrix(corr, z_frb, ell, f_sky=1.0):
    '''
    Plot the correlation coefficient matrix r_ij.
    '''
    N = len(z_frb)
    Ntot = N + 1

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot r_ij with symmetric color scale
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$r_{ij}$')

    labels = [f'$\\mathcal{{D}}(z={z:.2f})$' for z in z_frb] \
           + [f'$C_{{\\ell={ell}}}^{{\\mathcal{{DD}}}}$']
    ax.set_xticks(range(Ntot))
    ax.set_yticks(range(Ntot))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    # Highlight covariance blocks
    ax.add_patch(Rectangle((-0.5, -0.5), N, N,
                           fill=False, edgecolor='gold', lw=2,
                           label='$\\mathrm{Cov}[\\mathcal{D}_i, \\mathcal{D}_j]$'))

    ax.add_patch(Rectangle((N-0.5, -0.5), 1, N,
                           fill=False, edgecolor='deepskyblue', lw=2,
                           label='$\\mathrm{Cov}[\\mathcal{D}_i, C_\\ell]$'))
    ax.add_patch(Rectangle((-0.5, N-0.5), N, 1,
                           fill=False, edgecolor='deepskyblue', lw=2))

    ax.add_patch(Rectangle((N-0.5, N-0.5), 1, 1,
                           fill=False, edgecolor='blueviolet', lw=2,
                           label='$\\mathrm{Cov}[C_\\ell, C_\\ell]$'))

    ax.set_title(f'Correlation matrix ($\\ell={ell}$, $f_{{\\rm sky}}={f_sky}$)', fontsize=12)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.tick_params(which='major', direction='in', length=5, width=0.8, top=True, right=True)
    ax.tick_params(which='minor', direction='in', length=2, width=0.6, top=True, right=True)

    plt.tight_layout()
    plt.savefig(f'cov/correlation_matrix_ell{ell}.pdf', format="pdf", bbox_inches="tight")
    #plt.show()

    return fig
