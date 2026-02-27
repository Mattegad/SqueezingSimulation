import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# formule w(z)
def w_z(z, w0, z0, lam_nm):
    lam_mm = lam_nm * 1e-3  # convertir nm → um
    zR = np.pi * w0**2 / lam_mm  # Rayleigh length en um
    zR = zR * 1e-3  # convertir um → mm
    return w0 * np.sqrt(1.0 + ((z - z0)/zR)**2)

# Fonction de fit
def fit_beam_waist(z_data, w_data, lam_nm=852.0):
    w0_guess = np.min(w_data)
    z0_guess = z_data[np.argmin(w_data)]
    f = lambda z, w0, z0: w_z(z, w0, z0, lam_nm)

    popt, pcov = curve_fit(f, z_data, w_data, p0=[w0_guess, z0_guess])
    w0, z0 = popt
    w0_err, z0_err = np.sqrt(np.diag(pcov))
    return {'w0_um': w0, 'z0_mm': z0, 'w0_err_um': w0_err, 'z0_err_mm': z0_err,
            'fit_params': popt, 'cov': pcov}

# --------- NOUVEAU : plot X & Y sur la même figure ----------
def plot_fit_xy(z_data, w_data_x, w_data_y, lam_nm, fit_x, fit_y):
    w0x, z0x = fit_x
    w0y, z0y = fit_y

    z_dense = np.linspace(min(z_data), max(z_data), 500)
    w_fit_x = w_z(z_dense, w0x, z0x, lam_nm)
    w_fit_y = w_z(z_dense, w0y, z0y, lam_nm)

    plt.figure(figsize=(7,5))
    # raw data
    plt.scatter(z_data, w_data_x, color="red", label="Data X", s=35)
    plt.scatter(z_data, w_data_y, color="blue", label="Data Y", s=35)

    # fits
    plt.plot(z_dense, w_fit_x, color="red", linewidth=2,
             label=f"Fit X ($\omega_0$={w0x:.1f} µm, $z_0$={z0x:.1f} mm)")
    plt.plot(z_dense, w_fit_y, color="blue", linewidth=2,
             label=f"Fit Y ($\omega_0$={w0y:.1f} µm, $z_0$={z0y:.1f} mm)")

    plt.xlabel("z (mm)", fontsize=12)
    plt.ylabel("$\omega$ (µm)", fontsize=12)
    plt.title("Beam waist fit : X & Y", fontsize=14)
    plt.axhline(243, color='g', linewidth=2, linestyle='--', label="Cavity waist (243 µm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------
# Exemple d'utilisation
# -------------------------------
z_meas = np.array([200, 440, 540])
w_meas_x = 0.5*np.array([2200, 4100, 5000]) # µm
w_meas_y = 0.5*np.array([2200, 4100, 5000]) # µm


res_x = fit_beam_waist(z_meas, w_meas_x, lam_nm=426)
res_y = fit_beam_waist(z_meas, w_meas_y, lam_nm=426)

print("Fit X:", res_x)
print("Fit Y:", res_y)

# plot XY ensemble
plot_fit_xy(z_meas, w_meas_x, w_meas_y, 426,
            res_x["fit_params"], res_y["fit_params"])
