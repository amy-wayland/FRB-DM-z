import numpy as np
import os

from covariance import build_covariance_matrix, plot_correlation_matrix

if __name__ == "__main__":

    # ---------------------------------------------------
    # FRB sample
    # ---------------------------------------------------
    z_frb = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5])
    N = len(z_frb)
    f_sky = 1.0

    # ---------------------------------------------------
    # Sky geometry
    # ---------------------------------------------------
    rng = np.random.default_rng(42)
    ra = rng.uniform(0, 2*np.pi, N)
    dec = rng.uniform(-np.pi/2, np.pi/2, N)

    cos_theta = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            cos_theta[i, j] = (
                np.sin(dec[i]) * np.sin(dec[j]) +
                np.cos(dec[i]) * np.cos(dec[j]) * np.cos(ra[i] - ra[j]))

    # ---------------------------------------------------
    # Multipoles to compute
    # ---------------------------------------------------
    for ell in [10, 20, 50, 100, 200, 500]:
        save_file = f"cov/covariance_matrix_ell{ell}.npz"
        print("\n" + "=" * 50)
        print(f"Processing ell = {ell}")
        print("=" * 50)

        # ---------------------------------------------------
        # Load or compute
        # ---------------------------------------------------
        if os.path.exists(save_file):
            print(f"Loading existing file for ell={ell}...")
            data = np.load(save_file)
            cov = data["cov"]
            corr = data["corr"]

        else:
            print(f"Computing covariance matrix for ell={ell}...")
            
            cov, corr = build_covariance_matrix(
                ell,
                z_frb,
                cos_theta,
                f_sky=f_sky,
                Nchi=50,
                Nmu=40
            )

            os.makedirs(os.path.dirname(save_file), exist_ok=True)

            np.savez(
                save_file,
                cov=cov,
                corr=corr,
                z_frb=z_frb,
                cos_theta=cos_theta
            )

            print(f"Saved {save_file}")

        # ---------------------------------------------------
        # Plot
        # ---------------------------------------------------
        plot_correlation_matrix(corr, z_frb, ell, f_sky=f_sky)

        # ---------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------
        N = len(z_frb)

        print("\nCorrelation coefficient summary:")
        print(f"  Max |r| DD block:    {np.max(np.abs(corr[:N, :N] - np.eye(N))):.4f}")
        print(f"  Max |r| cross block: {np.max(np.abs(corr[:N, N])):.4f}")
        print(f"  Cov[C_ell, C_ell]:   {cov[N, N]:.4e}")
