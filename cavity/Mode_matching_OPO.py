
#%%
%matplotlib tk


#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys
from matplotlib.patches import Rectangle


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
w_cav = 200      # µm
z_cav_init = 1360      # mm
zR_cav = (np.pi * w_cav**2 / (lam0*1e-3))*1e-3  # mm
R = -2/(mirror_idx-1)*100  # mm rayon de courbure des miroirs en transmission
R = 1000000
l_cristal = 20  # mm
n_cristal = 1.84  # indice du cristal

# --- Paramètres du système ---
input_waist = 0.8  # mm
maxy = input_waist * 1000 * 4

foc1_init = 200  # mm
foc2_init = 150  # mm
pos1_init = 473  # mm
pos2_init = 1118  # mm
max_x = 2500  # mm
z0 = 0  # mm

# --- Fonctions ---
def round_to(xin, n_round):
    return round(xin, n_round)


# Rayleigh length
def zR(w):
    return pi * w**2 / (lam0 * 1e-6)


# Complex parameter
def q(z, w0, z0):
    return z-z0 + 1j * zR(w0)


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
    return Md(z7) @ Mf(R/2) @ Md(z6) @ RI(n_cristal,1) @ Md(z5) @ RI(1,n_cristal) @Md(z4) @ Mf(R/2) @ Md(z3) @ Mf(f2) @ Md(z2) @ Mf(f1) @ Md(z1) 


# q-propagation through each segment
def qprop1(z1, input_waist, l1, l2, foc1_init, foc2_init):
    return propagate(q(z=0, w0=input_waist, z0=z0), Mprop(z1, 0, 0, 0, 0, 0, 0, foc1_init, foc2_init))

def qprop2(z2, input_waist, l1, l2, foc1_init, foc2_init):
    return propagate(q(z=0, w0=input_waist, z0=z0), Mprop(l1, z2, 0, 0, 0, 0, 0, foc1_init, foc2_init))


def qprop3(z3, input_waist, l1, l2, foc1_init, foc2_init):
    return propagate(q(z=0, w0=input_waist, z0=z0), Mprop(l1, l2-l1, z3, 0, 0, 0, 0, foc1_init, foc2_init))


def qprop4(z4, input_waist, l1, l2, M1, foc1_init, foc2_init):
    return propagate(q(z=0, w0=input_waist, z0=z0), Mprop(l1, l2-l1, M1-l2, z4, 0, 0, 0, foc1_init, foc2_init))


def qprop5(z5, input_waist, l1, l2, M1, M2, foc1_init, foc2_init, pos_cristal):
    return propagate(q(z=0, w0=input_waist, z0=z0), Mprop(l1, l2-l1, M1-l2, pos_cristal-l_cristal/2-M1, z5, 0, 0, foc1_init, foc2_init))


def qprop6(z6, input_waist, l1, l2, M1, M2, foc1_init, foc2_init, pos_cristal):
    return propagate(q(z=0, w0=input_waist, z0=z0), Mprop(l1, l2-l1, M1-l2, pos_cristal-l_cristal/2-M1, l_cristal, z6, 0, foc1_init, foc2_init))


def qprop7(z7, input_waist, l1, l2, M1, M2, foc1_init, foc2_init, pos_cristal):
    return propagate(q(z=0, w0=input_waist, z0=z0), Mprop(l1, l2-l1, M1-l2, pos_cristal-l_cristal/2-M1, l_cristal, M2-(pos_cristal+l_cristal/2), z7, foc1_init, foc2_init))

# --- Fonction principale de propagation interactive ---
def propagation_interactive(l1, l2, z_cav, input_waist=input_waist):

    M1 = z_cav - 125/2  # mm
    M2 = M1 + 125  # mm
    M3 = M2 + 140  # mm
    M4 = M3 + 145  # mm
    M1bis = M4+140  # mm position miroir de repli
    pos_cristal = M1 + 125/2  # mm position du cristal

    z1_vals = np.linspace(0, l1, 300)
    z2_vals = np.linspace(0, l2-l1, 300)
    z3_vals = np.linspace(0, M1-l2, 1000)
    z4_vals = np.linspace(0, pos_cristal-l_cristal/2-M1, 300)
    z5_vals = np.linspace(0, l_cristal, 300)
    z6_vals = np.linspace(0, M2-(pos_cristal+l_cristal/2), 300)
    z7_vals = np.linspace(0, max_x, 300)

    r1 = [beam_radius(qprop1(z, input_waist, l1, l2, foc1_init, foc2_init)) 
          for z in z1_vals]
    r2 = [beam_radius(qprop2(z, input_waist, l1, l2, foc1_init, foc2_init)) 
          for z in z2_vals]
    r3 = [beam_radius(qprop3(z, input_waist, l1, l2, foc1_init, foc2_init))
           for z in z3_vals]
    r4 = [beam_radius(qprop4(z, input_waist, l1, l2, M1, foc1_init, foc2_init))
           for z in z4_vals]
    r5 = [beam_radius(qprop5(z, input_waist, l1, l2, M1, M2, foc1_init, foc2_init, pos_cristal))
           for z in z5_vals]
    r6 = [beam_radius(qprop6(z, input_waist, l1, l2, M1, M2, foc1_init, foc2_init, pos_cristal))
           for z in z6_vals]
    r7 = [beam_radius(qprop7(z, input_waist, l1, l2, M1, M2, foc1_init, foc2_init, pos_cristal))
           for z in z7_vals]

    z2_vals_shifted = z2_vals + l1
    z3_vals_shifted = z3_vals + l2
    z4_vals_shifted = z4_vals + M1
    z5_vals_shifted = z5_vals + pos_cristal - l_cristal/2
    z6_vals_shifted = z6_vals + pos_cristal + l_cristal/2
    z7_vals_shifted = z7_vals + M2

    # Trouver le waist dans le cristal
    idx_min = np.argmin(r5)
    waist_min = r5[idx_min]
    z_min = z5_vals_shifted[idx_min]

    # Trouver le waist entre les deux miroirs plans
    idx_min2 = np.argmin(r7)
    waist_min2 = r7[idx_min2]
    z_min2 = z7_vals_shifted[idx_min2]

    # Lignes verticales pour lentilles
    lens1_z = [l1, l1]
    lens2_z = [l2, l2]
    lens_y = [-2000, 2000]

    # Lignes verticales pour miroirs plans
    mirror3_z = [M3, M3]
    mirror4_z = [M4, M4]
    mirror_y = [-2000, 2000]

    # Lignes verticales pour miroirs courbes
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
    7: z7_vals_shifted
}

    r_vals_dict = {
    1: r1,
    2: r2,
    3: r3,
    4: r4,
    5: r5, 
    6: r6, 
    7: r7
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
    ax.text(l1,  2000, f"L1={foc1_init}", ha="center", va="bottom", fontsize=14)
    ax.text(l2,  2000, f"L2={foc2_init}", ha="center", va="bottom", fontsize=14)
    ax.text(M1,  2000, "M1", ha="center", va="bottom", fontsize=14)
    ax.text(M2,  2000, "M2", ha="center", va="bottom", fontsize=14)
    ax.text(M3,  2000, "M3", ha="center", va="bottom", fontsize=14)
    ax.text(M4,  2000, "M4", ha="center", va="bottom", fontsize=14)
    ax.text(M1bis, 2000, "M1'", ha="center", va="bottom", fontsize=14)

    ax.plot(z_min, waist_min, 'ko', markersize=8)
    ax.plot(z_min2, waist_min2, 'ko', markersize=8)
    ax.text(
    0.02, 0.95,
    f"W0_min = {waist_min:.1f}, {waist_min2:.1f} µm @ x = {z_min:.1f}, {z_min2:.1f} mm", 
    transform=ax.transAxes,
    fontsize=14,
    bbox=dict(facecolor='white', alpha=0.7))


    #Ticks and grid
    #ax.grid(False, which='both', linestyle='--', linewidth=0.5)
    ax.set_xticks(np.arange(0, max_x + 1, 20), minor=True)
    ax.set_yticks(np.arange(-3000, 3000, 50), minor=True)

    #ax.set_title("Propagation of a gaussian beam through two lenses")
    ax.set_xlabel("Distance (mm)", fontsize = 23)
    ax.set_ylabel("Beam waist (µm)", fontsize = 23)
    #ax.grid(True)
    ax.set_xlim(-100, max_x)
    #ax.set_ylim(-100,100)
    ax.set_ylim(-2000, 2000)
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

ax_l1 = fig.add_axes([0.15, 0.0, 0.8, 0.03])
ax_l2 = fig.add_axes([0.15, 0.05, 0.8, 0.03])
ax_z_cav = fig.add_axes([0.15, 0.1, 0.8, 0.03])
slider_l1 = Slider(ax_l1, 'Position L1 (mm)', 0, 1000, valinit=pos1_init, valstep=1)
slider_l2 = Slider(ax_l2, 'Position L2 (mm)', 50, 1300, valinit=pos2_init, valstep=0.5)
slider_z_cav = Slider(ax_z_cav, 'Longueur cavité (mm)', 0, 2000, valinit=z_cav_init, valstep=1)

def update(val):
    propagation_interactive(slider_l1.val, slider_l2.val, slider_z_cav.val)


slider_l1.on_changed(update)
slider_l2.on_changed(update)
slider_z_cav.on_changed(update)

propagation_interactive(slider_l1.valinit, slider_l2.valinit, slider_z_cav.valinit)
plt.show()


# %%
# ============================================================
# Optimisation conjointe f1, f2, l1, l2
# ============================================================

focal_list = np.array([100, 200, 250, 300])  # mm

l1_list = np.linspace(170, z_cav-200, 40)     # positions L1 (mm)
l2_list = np.linspace(170, z_cav-200, 40)    # positions L2 (mm)

z_search = np.linspace(0, max_x, 100)
alpha = 0  # poids pour la pénalité sur la position du waist

best_score = np.inf
best = None

q_in = q(z=0, w0=input_waist, z0=z0)

for f1 in focal_list:
    for f2 in focal_list:
        if f1 == f2:
            continue
        for l1 in l1_list:
            for l2 in l2_list:
                if l2 <= l1:
                    continue

                # propagation dans le cristal
                r_vals = []
                for z in z_search:
                    qz = propagate(
                        q_in, 
                        Mprop
                        (l1, 
                         l2-l1, 
                         M1-l2, 
                         pos_cristal-l_cristal/2-M1, 
                         z, 
                         0, 0, 
                         foc1_init, foc2_init
                         )
                    )
                    r_vals.append(beam_radius(qz))

                r_vals = np.array(r_vals)
                idx = np.argmin(r_vals)

                w_min = r_vals[idx]
                z_min = z_search[idx] + pos_cristal

                if abs(w_min - w_cav) > 10:
                    continue

                score = abs(w_min - w_cav) + alpha * z_min

                if score < best_score:
                    best_score = score
                    best = {
                        "f1": f1,
                        "f2": f2,
                        "l1": l1,
                        "l2": l2,
                        "w_min": w_min,
                        "z_min": z_min,
                        "score": score
                    }

# ============================================================
# Résultat
# ============================================================

print("=== OPTIMUM GLOBAL ===")
for k, v in best.items():
    if "w" in k:
        print(f"{k} = {v:.2f} µm")
    elif "z" in k or "l" in k:
        print(f"{k} = {v:.2f} mm")
    else:
        print(f"{k} = {v}")

# %%
# ============================================================
# MODE MATCHING OPO
# waist IMPOSE au centre de cavité z_cav
# ============================================================

target_waist = 45.0  # µm
tol_w = 10.0          # tolérance taille
tol_phase = 10     # tolérance condition de waist

alpha = 1e-2         # poids sur la condition de waist

focal_list = np.array([100, 200, 250, 300])
l1_list = np.linspace(300, 800, 50)
l2_list = np.linspace(400, 1100, 50)
z_cav_list = np.linspace(1000, 1800, 50)

best_score = np.inf
best = None

q_in = q(z=0, w0=input_waist, z0=z0)


for f1 in focal_list:
    for f2 in focal_list:
        if f1 == f2 and f1 != 300:
            continue

        for l1 in l1_list:
            for l2 in l2_list:
                if l2 <= l1:
                    continue

                for z_cav in z_cav_list:

                    # --- géométrie cavité ---
                    dM1M2 = 125  # mm distance entre les miroirs courbes
                    M1 = z_cav - dM1M2/2 # mm
                    pos_cristal = z_cav # mm position du cristal
                    q_cav = propagate(
                        q_in, 
                        Mprop
                        (l1, 
                         l2-l1, 
                         M1-l2, 
                         pos_cristal-l_cristal/2-M1, 
                         z_cav, 
                         0, 0, 
                         foc1_init, foc2_init
                         )
                    )
                    

                    # --- conditions de waist ---
                    inv_q = 1 / q_cav
                    waist_condition = abs(inv_q.real)
                    w_cav = beam_radius(q_cav)

                    if abs(w_cav - target_waist) > tol_w:
                        continue

                    if waist_condition > tol_phase:
                        continue

                    score = abs(w_cav - target_waist) + alpha * waist_condition

                    if score < best_score:
                        best_score = score
                        best = {
                            "f1": f1,
                            "l1": l1,
                            "f2": f2,
                            "l2": l2,
                            "z_cav": z_cav,
                            "w_cav": w_cav,
                            "Re(1/q)": inv_q.real,
                            "score": score
                        }

# ============================================================
# RÉSULTAT
# ============================================================

print("\n=== MODE MATCHING OPO OPTIMAL ===")
if best is None:
    print(
        "\n⚠️  Aucune solution trouvée.\n"
        "→ Aucune configuration (f1, l1, f2, l2, z_cav) ne satisfait simultanément :\n"
        "   • waist(z_cav) ≈ 45 µm\n"
        "   • condition de waist Re(1/q) ≈ 0 au centre de la cavité\n\n"
        "👉 Suggestions :\n"
        "   • élargir les plages de f1, f2, l1, l2 ou z_cav\n"
        "   • augmenter tol_w ou tol_phase\n"
        "   • vérifier la cohérence des paramètres de la cavité (R, L_cav)\n"
    )   
    raise SystemExit
for k, v in best.items():
    if "w" in k:
        print(f"{k} = {v:.2f} µm")
    elif "z" in k or "l" in k:
        print(f"{k} = {v:.2f} mm")
    else:
        print(f"{k} = {v:.3e}")

# %%
