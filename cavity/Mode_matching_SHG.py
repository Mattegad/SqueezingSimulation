
#%%
%matplotlib tk


#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys
from matplotlib.patches import Rectangle

delta_omega = 48.6 # Erreur sur waist pour être en dessous de 2% d'erreur sur g

sys.path.insert(0,'/Users/gadanimatteo/Documents/Stage:Thèse LKB/SqueezingSimulation')

from utils.settings import settings
import utils.plot_parameters as pm

# --- Constantes ---
lam0 = 852  # nm
c = 299.8   # nm/fs
pi = np.pi
mirror_size = 0.6 # mm
mirror_idx = 1.5 

# --- Paramètres cavité ---
w_cav = 252      # µm
z_cav = 880.9      # mm
zR_cav = (np.pi * w_cav**2 / (lam0*1e-3))*1e-3  # mm
R = 150  # mm rayon de courbure des miroirs
M1 = z_cav - 211/2  # mm
M2 = M1 + 211  # mm
M3 = M2 + 198  # mm
M4 = M3 + 183  # mm
M1bis = M4 + 198  # mm position miroir de repli
l_cristal = 20  # mm
n_cristal = 1.84  # indice du cristal
pos_cristal = M3 + 183/2  # mm position du cristal

# --- Paramètres du système ---
input_waist = 0.8  # mm
maxy = input_waist * 1000 * 4

foc1_init = 200  # mm
foc2_init = 125  # mm
pos1_init = 0  # mm
pos2_init = 352.5  # mm
max_x = 2500  # mm
zz = 0  # mm

# --- Fonctions ---
def round_to(xin, n_round):
    return round(xin, n_round)


# Rayleigh length
def zR(w):
    return pi * w**2 / (lam0 * 1e-6)


# Complex parameter
def q(z, w0):
    return z + 1j * zR(w0)


# Propagation through ABCD matrix
def propagate(confocal, mat):
    A, B = mat[0]
    C, D = mat[1]
    return (A * confocal + B) / (C * confocal + D)


# Waist
def waist(qparam):
    return np.sqrt(qparam.imag * lam0 / pi)


# Beam radius
def beam_radius(qparam):
    return np.sqrt(-(1 / (1 / qparam).imag) * lam0/ pi) # in um


# Matrix through a thin lense
def Mf_thin(foc):
    return np.array([[1.0, 0], [-1 / foc, 1.0]])

# Matrix through a through lense
def Mf(foc):
    n1, n2 = 1, mirror_idx
    A = np.array([[1.0, 0], [(n1-n2)/(n2*foc), n1/n2]])
    B = np.array([[1.0, 2.0], [0, 1.0]])
    C = np.array([[1.0, 0], [-(n2-n1)/(n1*foc), n2/n1]])
    return C @ B @ A


# Matrix through an interface
def RI(n1,n2) : 
    return np.array([[1,0],[0,n1/n2]])


# Matrix of free space travel
def Md(dist):
    return np.array([[1.0, dist], [0, 1.0]])





# Matrix of propagation
def Mprop(z1, z2, z3, z4, z5, z6, z7, f1, f2):#, n1, n2, mirror_size):
    return Md(z7) @ Mf(R/2) @ Md(z6) @ RI(n_cristal,1) @ Md(z5) @ RI(1,n_cristal) @ Md(z4) @ Mf(R/2) @ Md(z3) @ Mf(f2) @ Md(z2) @ Mf(f1) @ Md(z1) 


# q-propagation through each segment
def qprop1(z1, input_waist, l1, l2, foc1_init, foc2_init):
    return propagate(q(zz, input_waist), Mprop(z1, 0, 0, 0, 0, 0, 0, foc1_init, foc2_init))


def qprop2(z2, input_waist, l1, l2, foc1_init, foc2_init):
    return propagate(q(zz, input_waist), Mprop(l1, z2, 0, 0, 0, 0, 0, foc1_init, foc2_init))


def qprop3(z3, input_waist, l1, l2, foc1_init, foc2_init):
    return propagate(q(zz, input_waist), Mprop(l1, l2-l1, z3, 0, 0, 0, 0, foc1_init, foc2_init))


def qprop4(z4, input_waist, l1, l2, M3, foc1_init, foc2_init):
    return propagate(q(zz, input_waist), Mprop(l1, l2-l1, M3-l2, z4, 0, 0, 0, foc1_init, foc2_init))


def qprop5(z5, input_waist, l1, l2, M3, M4, foc1_init, foc2_init):
    return propagate(q(zz, input_waist), Mprop(l1, l2-l1, M3-l2, pos_cristal-l_cristal/2-M3, z5, 0, 0, foc1_init, foc2_init))


def qprop6(z6, input_waist, l1, l2, M3, M4, foc1_init, foc2_init):
    return propagate(q(zz, input_waist), Mprop(l1, l2-l1, M3-l2, pos_cristal-l_cristal/2-M3, l_cristal, z6, 0, foc1_init, foc2_init))


def qprop7(z7, input_waist, l1, l2, M3, M4, foc1_init, foc2_init):
    return propagate(q(zz, input_waist), Mprop(l1, l2-l1, M3-l2, pos_cristal-l_cristal/2-M3, l_cristal, M4-(pos_cristal+l_cristal/2), z7, foc1_init, foc2_init))


# --- Fonction principale de propagation interactive ---
def propagation_interactive(l1, l2, M3, M4, input_waist=input_waist):
    z1_vals = np.linspace(0, l1, 300)
    z2_vals = np.linspace(0, l2-l1, 300)
    z3_vals = np.linspace(0, M3-l2, 1000)
    z4_vals = np.linspace(0, pos_cristal-l_cristal/2-M3, 300)
    z5_vals = np.linspace(0, l_cristal, 300)
    z6_vals = np.linspace(0, M4-(pos_cristal+l_cristal/2), 300)
    z7_vals = np.linspace(0, max_x, 300)
    #z3_vals = np.linspace(0, max_x - (l1 + l2), 300)

    r1 = [beam_radius(qprop1(z, input_waist, l1, l2, foc1_init, foc2_init)) 
          for z in z1_vals]
    r2 = [beam_radius(qprop2(z, input_waist, l1, l2, foc1_init, foc2_init)) 
          for z in z2_vals]
    r3 = [beam_radius(qprop3(z, input_waist, l1, l2, foc1_init, foc2_init))
           for z in z3_vals]
    r4 = [beam_radius(qprop4(z, input_waist, l1, l2, M3, foc1_init, foc2_init))
           for z in z4_vals]
    r5 = [beam_radius(qprop5(z, input_waist, l1, l2, M3, M4, foc1_init, foc2_init))
           for z in z5_vals]
    r6 = [beam_radius(qprop6(z, input_waist, l1, l2, M3, M4, foc1_init, foc2_init))
           for z in z6_vals]
    r7 = [beam_radius(qprop7(z, input_waist, l1, l2, M3, M4, foc1_init, foc2_init))
           for z in z7_vals]

    z2_vals_shifted = z2_vals + l1
    z3_vals_shifted = z3_vals + l2
    z4_vals_shifted = z4_vals + M3
    z5_vals_shifted = z5_vals + pos_cristal - l_cristal/2
    z6_vals_shifted = z6_vals + pos_cristal + l_cristal/2
    z7_vals_shifted = z7_vals + M4

    # --- trouver waist min après l2 ---
    idx_min = np.argmin(r3)
    waist_min = r3[idx_min]
    z_min = z3_vals_shifted[idx_min]

    # --- trouver waist min après M3 ---
    idx_min2 = np.argmin(r5)
    waist_min2 = r5[idx_min2]
    z_min2 = z5_vals_shifted[idx_min2]


    # Calcul du mode matching
    Cx = (2 * w_cav * waist_min)/(w_cav**2 + waist_min**2) * np.sqrt(1/(1 + ((z_min - z_cav)/zR_cav)**2))

    # Lignes verticales pour lentilles
    lens1_z = [l1, l1]
    lens2_z = [l2, l2]
    lens_y = [-input_waist * 1000, input_waist * 1000]

    # Lignes verticales pour miroirs courbés
    mirror3_z = [M3, M3]
    mirror4_z = [M4, M4]
    mirror_y = [-input_waist * 1000, input_waist * 1000]

    # Lignes verticales pour miroirs droits
    mirror5_z = [M1bis, M1bis]
    mirror1_z = [M1, M1]
    mirror2_z = [M2, M2]

    z_vals_dict = {
    1: z1_vals,
    2: z2_vals_shifted,
    3: z3_vals_shifted,
    4: z4_vals_shifted,
    5: z5_vals_shifted,
    6: z6_vals_shifted,
    7: z7_vals_shifted,
}

    r_vals_dict = {
    1: r1,
    2: r2,
    3: r3,
    4: r4,
    5: r5,
    6: r6,
    7: r7,
}
    # --- Affichage du rayon r(z) au niveau de chaque optique ---

    optic_positions = {
    "L1": l1,
    "L2": l2,
    "M1": M1,
    "M2": M2,
    "M3": M3,
    "M4": M4,
    "M1'": M1bis
}

    optic_radii = {
    "L1": r1[-1],
    "L2": r2[-1],
    "M1": r3[int((M1-l2)/(M3-l2)*1000)],  # r3 à la position de M1 = r3[(M1 - l2)/ (M3 - l2)]
    "M2": r3[int((M2-l2)/(M3-l2)*1000)],  # r3 à la position de M2 = r3[(M2 - l2)/ (M3 - l2)]
    "M3": r3[-1],
    "M4": r6[-1],
    "M1'": r7[int((M1bis-M4)/ (max_x)*(300-1))]  # r7 à la position de M1' = r7[(M1' - M4)/ (max_x) * (300-1)]
}


    # --- Mise à jour du plot ---
    ax.clear()
    for i in range(1, 8):
        z_vals = z_vals_dict[i]
        r_vals = r_vals_dict[i]

        ax.plot(z_vals,  r_vals,  'r', linewidth=1)
        ax.plot(z_vals, [-v for v in r_vals], 'r', linewidth=1)


    ax.plot(lens1_z, lens_y, 'b--', linewidth=1.5, label=f'L1 @ {l1} mm (f={foc1_init} mm)')
    ax.plot(lens2_z, lens_y, 'b--', linewidth=1.5, label=f'L2 @ {l1 + l2} mm (f={foc2_init} mm)')
    ax.plot(mirror1_z, mirror_y, 'y--', linewidth=1.5, label=f'M3 @ {M3} mm (R={R} mm)')
    ax.plot(mirror2_z, mirror_y, 'y--', linewidth=1.5, label=f'M4 @ {M4} mm (R={R} mm)')
    ax.plot(mirror3_z, mirror_y, 'g--', linewidth=1.5, label=f'M4 @ {M4} mm (R={R} mm)')
    ax.plot(mirror4_z, mirror_y, 'g--', linewidth=1.5, label=f'M4 @ {M4} mm (R={R} mm)')
    ax.plot(mirror5_z, mirror_y, 'y--', linewidth=1.5, label=f'M4 @ {M4} mm (R={R} mm)')

    # --- Annotations des composants ---
    ax.text(l1,  input_waist*1000, "L1", ha="center", va="bottom", fontsize=14)
    ax.text(l2,  input_waist*1000, "L2", ha="center", va="bottom", fontsize=14)

    ax.text(M1,  input_waist*1000, "M1", ha="center", va="bottom", fontsize=14)
    ax.text(M2,  input_waist*1000, "M2", ha="center", va="bottom", fontsize=14)
    ax.text(M3,  input_waist*1000, "M3", ha="center", va="bottom", fontsize=14)
    ax.text(M4,  input_waist*1000, "M4", ha="center", va="bottom", fontsize=14)
    ax.text(M1bis, input_waist*1000, "M1'", ha="center", va="bottom", fontsize=14)

    # affichage texte
    for label in optic_radii:
        z_pos = optic_positions[label]
        r_val = optic_radii[label]
        ax.text(
        z_pos, -15, f"{r_val:.1f} µm",
        ha="center", va="top",
        fontsize=9
    )

    ax.plot(z_min, waist_min, 'ko', markersize=8)
    ax.plot(z_min2, waist_min2, 'mo', markersize=8)
    ax.text(
    0.02, 0.95,
    f"W0_min = {waist_min:.1f}, {waist_min2:.1f} µm @ x = {z_min:.1f}, {z_min2:.1f} mm, Mode-matching η = {Cx:.3f}", 
    transform=ax.transAxes,
    fontsize=14,
    bbox=dict(facecolor='white', alpha=0.7))
    #Ticks and grid
    #ax.grid(False, which='both', linestyle='--', linewidth=0.5)
    ax.set_xticks(np.arange(0, max_x + 1, 20), minor=True)
    ax.set_yticks(np.arange(-1200, 1201, 50), minor=True)

    #ax.set_title("Propagation of a gaussian beam through two lenses")
    ax.set_xlabel("Distance (mm)", fontsize = 23)
    ax.set_ylabel("Beam waist (µm)", fontsize = 23)
    #ax.grid(True)
    ax.set_xlim(-100, max_x)
    #ax.set_ylim(-100,100)
    ax.set_ylim(-input_waist*1500, input_waist*1500)
    ax.axhline(w_cav, linewidth = 1)
    ax.axhline(-w_cav, linewidth = 1)
    #ax.axhline(w_cav+delta_omega, linewidth = 1, color='orange')
    #ax.axhline(-w_cav-delta_omega, linewidth = 1, color='orange')
    #ax.axhline(w_cav-delta_omega, linewidth = 1, color='orange')
    #ax.axhline(-w_cav+delta_omega, linewidth = 1, color='orange')
    #ax.axvline(820, linewidth = 1, color='g')
    #ax.axvline(920, linewidth = 1, color='g')
    #ax.legend()

    

    # --- Petite box autour du cristal ---
    box_width  = l_cristal                     # largeur = longueur du cristal
    box_height = 1000                            # hauteur totale
    box_x = pos_cristal - box_width / 2        # coin bas-gauche X
    box_y = -box_height / 2                    # coin bas-gauche Y (centrée verticalement)

    crystal_box = Rectangle(
    (box_x, box_y),
    box_width,
    box_height,
    edgecolor='black',
    facecolor='none',
    linewidth=1.5
)

    ax.add_patch(crystal_box)

    fig.canvas.draw_idle()


# --- Configuration de la figure et des sliders ---
fig, ax = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(bottom=0.25, left=0.1, right=0.95, top=0.9)

ax_l1 = fig.add_axes([0.15, 0.05, 0.8, 0.03])
ax_l2 = fig.add_axes([0.15, 0.1, 0.8, 0.03])

slider_l1 = Slider(ax_l1, 'Position L1 (mm)', 0, 400, valinit=pos1_init, valstep=1)
slider_l2 = Slider(ax_l2, 'Position L2 (mm)', 50, 500, valinit=pos2_init, valstep=0.5)


def update(val):
    propagation_interactive(slider_l1.val, slider_l2.val, M3, M4)


slider_l1.on_changed(update)
slider_l2.on_changed(update)

propagation_interactive(slider_l1.valinit, slider_l2.valinit, M3, M4)
plt.show()
# %%

# Compute the output waist position vs input waist position through the two lenses system

z_waist_in = np.linspace(-2000, 2000, 1000)
q_in = q(z_waist_in, input_waist)
Mat = Mf(125) @ Md(352.5) @ Mf(200)
a, b = Mat[0]
c, d = Mat[1]
z_waist_out = np.array([-np.real((a * q + b) / (c * q + d)) for q in q_in])
plt.plot(z_waist_in, z_waist_out)
plt.xlabel("Input waist position (mm)")
plt.ylabel("Output waist position (mm)")
plt.title("Output waist position vs Input waist position through the two lenses system")
plt.grid()
plt.show()
# %%


# Compute the output waist position vs distance between the lenses through the two lenses system

dd = np.linspace(50, 500, 1000)
q_in = q(0, input_waist)
z_waist_out = []
for d in dd:
    Mat = Mf(125) @ Md(d) @ Mf(200)
    a, b = Mat[0]
    c, d = Mat[1]
    z_waist_out.append(-np.real((a * q_in + b) / (c * q_in + d)))
     
plt.plot(dd, z_waist_out)
plt.xlabel("Distance between the lenses (mm)")
plt.ylabel("Output waist position (mm)")
plt.title("Output waist position vs Input waist position through the two lenses system")
plt.grid()
plt.show()

# %%

# Combine the two previous plots to find the output waist position vs input waist position and distance between the lenses

# grilles à balayer
z_waist_in_vals = np.linspace(-2000, 2000, 400)   # mm
d_vals = np.linspace(200, 500, 400)                # mm



# matrice résultat : z_out[i,j] = waist out pour z_in[i] et d[j]
z_waist_out_map = np.zeros((len(z_waist_in_vals), len(d_vals)))

# ===== double boucle =====
for i, z_in in enumerate(z_waist_in_vals):
    q_in = q(z_in, input_waist)

    for j, dist in enumerate(d_vals):

        # Matrice totale : L2 → espace dist → L1
        M = Mf(125) @ Md(dist) @ Mf(200)
        a, b = M[0]
        c, d = M[1]

        q_out = (a*q_in + b) / (c*q_in + d)
        z_out = -np.real(q_out)

        z_waist_out_map[i, j] = z_out

# ==== Plot ====
plt.figure(figsize=(10,6))
plt.imshow(z_waist_out_map, 
           extent=[d_vals[0], d_vals[-1], z_waist_in_vals[0], z_waist_in_vals[-1]],
           aspect='auto',
           origin='lower',
           cmap='turbo')

plt.colorbar(label="Output waist position (mm)")
plt.xlabel("Lens separation d (mm)")
plt.ylabel("Input waist position (mm)")
plt.title("Output waist position vs input waist & lens separation")

z_target = 524   # mm
delta_z  = 0.05*z_target    # mm width of the band (ex: 524 ± 20)

# Niveau exact
#plt.contour(d_vals, z_waist_in_vals, z_waist_out_map, 
            #levels=[z_target], colors='white', linewidths=1.8)

# Bande autour de la valeur : niveaux z_target ± delta_z
plt.contour(d_vals, z_waist_in_vals, z_waist_out_map, 
            levels=[z_target-delta_z, z_target+delta_z], 
            colors='white', linestyles='dashed', linewidths=1.2)

plt.show()

# %%
# grilles à balayer
z_waist_in_vals = np.linspace(-2000, 2000, 400)   # mm
d_vals = np.linspace(200, 500, 400)                # mm

# matrice résultat : z_out[i,j] = waist out pour z_in[i] et d[j]
r_waist_out_map = np.zeros((len(z_waist_in_vals), len(d_vals)))

# ===== double boucle =====
for i, z_in in enumerate(z_waist_in_vals):
    q_in = q(z_in, input_waist)

    for j, dist in enumerate(d_vals):

        # Matrice totale : L2 → espace dist → L1
        M = Mf(125) @ Md(dist) @ Mf(200)
        a, b = M[0]
        c, d = M[1]

        q_out = (a*q_in + b) / (c*q_in + d)
        z_out = -np.real(q_out)
        r_out = beam_radius(q_out+z_out)
        r_waist_out_map[i, j] = r_out

# ==== Plot ====
plt.figure(figsize=(10,6))
plt.imshow(r_waist_out_map, 
           extent=[d_vals[0], d_vals[-1], z_waist_in_vals[0], z_waist_in_vals[-1]],
           aspect='auto',
           origin='lower',
           cmap='turbo')

plt.colorbar(label="waist radius (µm)")
plt.xlabel("Lens separation d (mm)")
plt.ylabel("Input waist position (mm)")
plt.title("Waist vs input waist & lens separation")
w_target = 250   # µm
delta_w  = 0.05*w_target    # µm width around the target (250 ± 20)

# Niveau exact
#plt.contour(d_vals, z_waist_in_vals, r_waist_out_map, 
#            levels=[w_target], colors='white', linewidths=1.8)

# Bande ± delta_w
plt.contour(d_vals, z_waist_in_vals, r_waist_out_map, 
            levels=[w_target-delta_w, w_target+delta_w], 
            colors='white', linestyles='dashed', linewidths=1.2)
plt.contour(d_vals, z_waist_in_vals, z_waist_out_map, 
            levels=[z_target-delta_z, z_target+delta_z], 
            colors='black', linestyles='dashed', linewidths=1.2)

plt.show()

# %%

z_target = 524   # mm
delta_z  = 0.05*z_target    # mm width of the band (ex: 524 ± 20)
w_target = 250   # µm
delta_w  = 0.05*w_target    # µm width around the target (250 ± 20)
input_waist_val = np.linspace(0.7, 0.9, 400)    # mm
d_vals = np.linspace(200, 500, 400)                # mm



# matrice résultat : z_out[i,j] = waist out pour z_in[i] et d[j]
z_waist_out_map = np.zeros((len(input_waist_val), len(d_vals)))
r_waist_out_map = np.zeros((len(input_waist_val), len(d_vals)))

# ===== double boucle =====
for i, w_in in enumerate(input_waist_val):
    q_in = q(0, w_in)

    for j, dist in enumerate(d_vals):

        # Matrice totale : L2 → espace dist → L1
        M = Mf(125) @ Md(dist) @ Mf(200)
        a, b = M[0]
        c, d = M[1]

        q_out = (a*q_in + b) / (c*q_in + d)
        z_out = -np.real(q_out)

        z_waist_out_map[i, j] = z_out

        r_out = beam_radius(q_out+z_out)
        r_waist_out_map[i, j] = r_out

# ==== Plot ====
plot = {"Radius": r_waist_out_map, "Position": z_waist_out_map}
contour = {"Radius": w_target, "Position": z_target}
delta_contour = {"Radius": delta_w, "Position": delta_z}
colorlabel = {"Radius": "Waist radius (µm)", "Position": "Output waist position (mm)"}
which_plot = "Position"  # "Radius" or "Position"
which_contour = contour[which_plot]
which_colorlabel = colorlabel[which_plot]
which_delta = delta_contour[which_plot]


plt.figure(figsize=(10,6))
plt.imshow(plot[which_plot], 
           extent=[d_vals[0], d_vals[-1], input_waist_val[0], input_waist_val[-1]],
           aspect='auto',
           origin='lower',
           cmap='turbo')

plt.colorbar(label=which_colorlabel)
plt.xlabel("Lens separation d (mm)")
plt.ylabel("Input waist (mm)")




# Niveau exact
#plt.contour(d_vals, z_waist_in_vals, z_waist_out_map, 
            #levels=[z_target], colors='white', linewidths=1.8)

# Bande autour de la valeur : niveaux z_target ± delta_z
plt.contour(d_vals, input_waist_val, plot[which_plot], 
            levels=[which_contour-which_delta, which_contour+which_delta], 
            colors='white', linestyles='dashed', linewidths=1.2)

plt.contour(d_vals, input_waist_val, plot["Radius"], 
            levels=[250-delta_w, 250+delta_w], 
            colors='black', linestyles='dashed', linewidths=1.2)

plt.show()
# %%
