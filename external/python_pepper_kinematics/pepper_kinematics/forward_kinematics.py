import math
import numpy as np
import scipy as sp
from scipy import linalg


joint_distances = np.array([
 0.14974,
 0.015,
 0.1812,
 0,
 0.150,
 0.0695,
 0.0303
])

p = np.array([0,0,0,1])
v0 = np.array([[1],[0],[0],[0]])
v1 = np.array([[0],[1],[0],[0]])
v2 = np.array([[0],[0],[1],[0]])


def transX(th, x, y, z):
    s = math.sin(th)
    c = math.cos(th)
    return np.array([[1, 0, 0, x], [0, c, -s, y], [0, s, c, z], [0, 0, 0, 1]])

def transY(th, x, y, z):
    s = math.sin(th)
    c = math.cos(th)
    return np.array([[c, 0, -s, x], [0, 1, 0, y], [s, 0, c, z], [0, 0, 0, 1]])

def transZ(th, x, y, z):
    s = math.sin(th)
    c = math.cos(th)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def calc_fk_and_jacob(angles, jacob=True, right=True, scale=1., full_pos=False):
    L1, L2, L3, L4, L5, L6, L7 = joint_distances * scale
    _L1_ = -L1 if right else L1
    _L2_ = -L2 if right else L2

    # s: shoulder (z rotated away from torso)
    t_T_s = transY(-angles[0], 0, _L1_, 0)

    # as: arm (shoulder end / bicep), ae: arm (elbow end), e: elbow,
    # fn: forearm but wrong orientation in left arm, f: forearm, w: wrist, h: hand
    s_T_as = transZ(angles[1], 0, 0, 0)
    as_T_ae = transY(9.0/180.0*math.pi, L3, _L2_, 0)
    ae_T_e = transX(angles[2]+math.pi/2., 0, 0, 0)
    e_T_fn = transZ(angles[3] if right else -angles[3], 0, 0, 0)
    fn_T_f = transX(0 if right else math.pi, 0, 0, 0)
    f_T_w = transX(angles[4], L5, 0, 0)
    w_T_h = transZ(0, L6, 0, -L7)

    t_T_as = t_T_s.dot(s_T_as)
    t_T_ae = t_T_as.dot(as_T_ae)
    t_T_e = t_T_ae.dot(ae_T_e)
    t_T_fn = t_T_e.dot(e_T_fn)
    t_T_f = t_T_fn.dot(fn_T_f)
    t_T_w = t_T_f.dot(f_T_w)
    t_T_h = t_T_w.dot(w_T_h)

    pos = t_T_h.dot(p)[:3]
    ori = t_T_h[0:3,0:3]

    if full_pos:
        pos = [T.dot(p) for T in [t_T_s, t_T_as, t_T_e, t_T_f, t_T_w, t_T_h]]
        ori = [T[0:3,0:3] for T in [t_T_s, t_T_as, t_T_e, t_T_f, t_T_w, t_T_h]]

    if not jacob:
        return pos, ori

    OfstT1 = _L1_ * t_T_s.dot(v1)
    OfstTd = t_T_ae.dot(np.array([[L3], [_L2_], [0], [0]]))
    OfstT5 = L5 * t_T_w.dot(v0)
    OfstT6 = t_T_w.dot(np.array([[L6], [0], [-L7], [0]]))

    vec6 = OfstT6
    vec5 = vec6 + OfstT5
    vec4 = vec5
    vec3 = vec4
    vecd = vec3 + OfstTd
    vec2 = vecd
    vec1 = vec2 + OfstT1

    j1 = t_T_s.dot(v1)
    j2 = t_T_as.dot(v2)
    jd = t_T_ae.dot(v1)
    j3 = t_T_e.dot(v0)
    j4 = t_T_f.dot(v2)
    j5 = t_T_w.dot(v0)

    J1 = cross(j1, vec1)
    J2 = cross(j2, vec2)
    J3 = cross(j3, vec3)
    J4 = cross(j4, vec4)
    J5 = cross(j5, vec5)

    J = np.c_[J1, J2, J3, J4, J5]
    return pos, ori, J


def cross(j, v):
    t0 = j[1][0] * v[2][0] - j[2][0] * v[1][0]
    t1 = j[2][0] * v[0][0] - j[0][0] * v[2][0]
    t2 = j[0][0] * v[1][0] - j[1][0] * v[0][0]
    return np.array([[t0], [t1], [t2]])

