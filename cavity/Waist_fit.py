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
z_meas = 25.1*np.array([3, 5, 7, 9, 11, 13, 15, 18, 21])  # mm
z_meas2 = 25.1*np.array([5, 7, 9, 11, 13, 15, 18, 21])  # mm
w_meas_x = 0.5*np.array([670, 620, 550, 500, 480, 475, 490, 510, 575]) # µm
w_meas_y = 0.5*np.array([680, 610, 500, 480, 465, 460, 480, 520, 580]) # µm
w_meas_x2 = 0.5*np.array([645, 605, 540, 505, 500, 510, 565, 590]) # µm
w_meas_y2 = 0.5*np.array([635, 550, 515, 495, 485, 505, 565, 600]) # µm

res_x = fit_beam_waist(z_meas2, w_meas_x2, lam_nm=852)
res_y = fit_beam_waist(z_meas2, w_meas_y2, lam_nm=852)

print("Fit X:", res_x)
print("Fit Y:", res_y)

# plot XY ensemble
plot_fit_xy(z_meas2, w_meas_x2, w_meas_y2, 852,
            res_x["fit_params"], res_y["fit_params"])
