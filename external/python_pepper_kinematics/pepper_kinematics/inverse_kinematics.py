import math
import numpy as np
import scipy as sp
from scipy import linalg

import forward_kinematics as fk
from joint_definitions import clamp_joints

def calc_inv_pos(angles, target_pos, target_ori, epsilon, right=True):
    p  = np.array([0,0,0,1])
    angs = np.array([a for a in angles])
    sum_old = 100000
    MAX_ITER = 1000
    for i in range(MAX_ITER):
        pos, ori, j = fk.calc_fk_and_jacob(angs, jacob=True, right=right)
        J = _calc_invJ(j)
        delta_pos = np.matrix((target_pos-pos)[0:3]).transpose()
        v = (J * delta_pos).transpose()
        angs = np.squeeze(np.asarray(v)) + angs
        
        sum = 0
        for d in delta_pos:
            sum = sum + math.fabs(d)
        #sum = np.sum(delta_pos)
        if sum < epsilon:
            break
        if sum > sum_old:
#             print '# set_position error : Distance can not converged.'
            return None
        sum_old = sum
    if i == MAX_ITER - 1:
        print("Max number of iterations reached!")
    return angs

def single_step_towards_target(angles, target_pos, target_ori, scale=1., max_delta=0.1, right=True):
    angs = np.array(angles)
    pos, ori, j = fk.calc_fk_and_jacob(angs, jacob=True, scale=scale, right=right)
    # follow jacobian towards new joint limits
    J = _calc_invJ(j)
    delta_pos = np.matrix((target_pos-pos)[0:3]).transpose()
    # cap the position difference vector
    delta = np.linalg.norm(delta_pos)
    if delta > max_delta:
        delta_pos = max_delta * delta_pos / delta
    delta_angs = (J * delta_pos).transpose()
    angs = np.squeeze(np.asarray(delta_angs)) + angs
    # respect joint limits
    angs = clamp_joints(angs, right=right)
    # did we get closer to desired? if not, singularity, return no movement
    new_pos, new_ori = fk.calc_fk_and_jacob(angs, jacob=False, scale=scale, right=right)
    if np.linalg.norm(new_pos-target_pos) >= np.linalg.norm(pos-target_pos):
        return np.array(angles)
    # return new angles
    return angs

def _calc_invJ(J, epsilon = 0.01):
    u, sigma, v = np.linalg.svd(J, full_matrices=0)
    sigma_ = [1/s if s > epsilon else 0 for s in sigma]
    rank = np.shape(J)[0]
    return np.matrix(v.transpose()) * np.matrix(linalg.diagsvd(sigma_, rank, rank)) * np.matrix(u.transpose())
