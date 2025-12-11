# -*- coding: utf-8 -*-
"""
Cavité bow-tie avec :
 - M1, M2 plans
 - M3, M4 courbes (R)
 - Un cristal d’indice n au centre de M3–M4
 - Calcul du mode propre (q_auto-cohérent)
 - Deux waists automatiquement visibles
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

pi = np.pi
lam0 = 852.0                    # nm
lam_mm = lam0 * 1e-6           # mm


# -----------------------------
#  ABCD UTILITIES
# -----------------------------
def Md(d):
    """Propagation libre en mm"""
    return np.array([[1, d], [0, 1]])

def Mf(f):
    """Matrice miroir courbe / lentille fine"""
    return np.array([[1, 0], [-1/f, 1]])

def refract_interface(n1, n2):
    """Interface plane n1 -> n2 (plan-plan)"""
    return np.array([[1, 0], [0, n1/n2]])

def M_medium(d, n):
    """
    Propagation dans un milieu d’indice n.
    zR est compressé par n.
    """
    return np.array([[1, d/n], [0, 1]])


def propagate(q, M):
    A,B = M[0]
    C,D = M[1]
    return (A*q + B) / (C*q + D)


# -----------------------------
#  MODE PROPRE D’UNE CAVITÉ
# -----------------------------
def q_eigenmode(A,B,C,D):
    """
    Résout q = (Aq+B)/(Cq+D)  → q eigenmode
    q = (A-D)/(2C) ± sqrt((A+D)^2 -4)/(2C)
    On choisit Im(q)>0
    """
    trace = A + D
    disc = trace**2 - 4
    root = np.sqrt(disc + 0j)     # complexe

    q1 = ( (A-D) + root ) / (2*C)
    q2 = ( (A-D) - root ) / (2*C)

    return q1 if q1.imag > 0 else q2


# -----------------------------
#  CALCUL DU TOUR COMPLET
# -----------------------------
def M_round_trip(d12, d23, d34, d41, R, Lc, n):
    """
    Construction du round trip :
    M1(p) -> d12 -> M2(p) -> d23 -> M3(c) -> cristal -> M4(c) -> d41 -> retour M1
    """

    f = R/2

    # Miroirs
    Mplan = np.eye(2)
    Mcurve = Mf(f)

    # Segments & cristal
    # M3->M4 contient le cristal centré
    L_air = (d34 - Lc)/2

    # Matrice totale (ordre : M4 -> M1 -> M2 -> M3 -> M4)
    M = np.eye(2)

    # 1) M4 -> M1
    M = Md(d41) @ M

    # 2) M3 -> M4
    M = Mcurve @ M
    M = M_medium(L_air, 1) @ M
    M = refract_interface(1, n) @ M
    M = M_medium(Lc, n) @ M
    M = refract_interface(n, 1) @ M
    M = M_medium(L_air, 1) @ M

    # 3) M2 -> M3
    M = Md(d23) @ M
    M = Mcurve @ M

    # 4) M1 -> M2
    M = Md(d12) @ M
    M = Mplan @ M

    return M


# -----------------------------
#  PROPAGATION LINEAIRE
# -----------------------------
def propagate_along_cavity(q0, d12, d23, d34, d41, R, Lc, n, N=300):
    xs = []
    ws = []

    f = R/2

    # positions
    pos = 0

    def push_segment(d, q):
        """propagation dans air"""
        nonlocal pos
        xloc = np.linspace(0,d,N)
        for x in xloc:
            qx = propagate(q, Md(x))
            xs.append(pos + x)
            ws.append( np.sqrt(lam_mm/pi * qx.imag) )
        q_end = propagate(q, Md(d))
        pos += d
        return q_end

    def push_crystal(d, q):
        """propagation dans le cristal (indice n)"""
        nonlocal pos
        xloc = np.linspace(0,d,N)
        for x in xloc:
            qx = propagate(q, M_medium(x, n))
            xs.append(pos + x)
            ws.append( np.sqrt(lam_mm/pi * qx.imag) )
        q_end = propagate(q, M_medium(d, n))
        pos += d
        return q_end

    # chemin complet M1 -> M2 -> M3 -> Cristal -> M4 -> M1

    # M1->M2
    q = q0
    q = push_segment(d12, q)

    # M2->M3
    q = push_segment(d23, q)

    # M3 miroir courbe
    q = propagate(q, Mf(f))

    # segment M3->M4
    L_air = (d34 - Lc)/2
    q = push_segment(L_air, q)

    # entrée cristal
    q = propagate(q, refract_interface(1,n))
    q = push_crystal(Lc, q)
    q = propagate(q, refract_interface(n,1))

    # sortie cristal
    q = push_segment(L_air, q)

    # M4 miroir courbe
    q = propagate(q, Mf(f))

    # M4->M1
    q = push_segment(d41, q)

    return np.array(xs), np.array(ws)


# -----------------------------
#  PARAMÈTRES INITIAUX
# -----------------------------
d12 = 300
d23 = 200
d34 = 150
d41 = 350
R = 200
Lc = 10
n = 1.8     # indice du cristal (KTP typiquement)


# -----------------------------
#  CALCUL MODE PROPRE
# -----------------------------
M = M_round_trip(d12, d23, d34, d41, R, Lc, n)
A,B = M[0]
C,D = M[1]
q0 = q_eigenmode(A,B,C,D)


# -----------------------------
#  TRACÉ
# -----------------------------
xs, ws = propagate_along_cavity(q0, d12, d23, d34, d41, R, Lc, n)

plt.figure(figsize=(13,6))
plt.plot(xs, ws*1000, 'r')
plt.plot(xs, -ws*1000, 'r')
plt.xlabel("Position le long de la cavité (mm)")
plt.ylabel("Rayon du faisceau (µm)")
plt.grid(True)

# marqueurs miroirs
plt.axvline(d12, color='k', linestyle='--')               # M2
plt.axvline(d12+d23, color='k', linestyle='--')           # M3
plt.axvline(d12+d23+d34, color='k', linestyle='--')       # M4

# cristal
xM3 = d12+d23
plt.axvspan(xM3 + (d34-Lc)/2, xM3 + (d34-Lc)/2 + Lc,
            color='cyan', alpha=0.3, label='Cristal')

plt.legend()
plt.show()

# %%
