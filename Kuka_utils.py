import crocoddyl
import pinocchio as pin
from IPython.display import HTML
import mim_solvers
import hppfcl
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from mim_robots.robot_loader import load_bullet_wrapper, load_mujoco_model, get_robot_list, load_pinocchio_wrapper

class obstacle_set():
    def __init__(self):
        self.obs_num = 0
        self.obs_p = []
        self.obs_l = []
        self.col_r = []
    def add_obs(self, p, l, col_r):
        self.obs_num += 1
        self.obs_p.append(p)
        self.obs_l.append(l)
        self.col_r.append(col_r)

def create_obs(pin_model, pos, rot, m, d, num_obs):
    obsPose = pin.SE3.Identity()
    obsPose.rotation = rot
    obsPose.translation = pos
    obsObj = pin.GeometryObject("obstacle"+str(num_obs),
                                m.getFrameId("universe"),
                                m.frames[pin_model.getFrameId("universe")].parent,
                                hppfcl.Box(d[0], d[1], d[2]),
                                obsPose)
    return obsObj

def convert_2_cart(states, frame_id):
    size = states.shape
    cart_pred = np.zeros(shape = (size[0], size[1], 3))
    dummy_robot = load_pinocchio_wrapper("iiwa")
    m = dummy_robot.model
    d = dummy_robot.data
    for cycle in range(size[0]):
        for t in range(size[1]):
            q = states[cycle,t,:m.nq]
            v = states[cycle,t,m.nq:]
            pin.forwardKinematics(m,d,q)
            pin.framesForwardKinematics(m, d, q)
            pin.updateFramePlacements(m, d)  
            p = d.oMf[frame_id].copy()
            cart_pred[cycle,t,:] = p.translation

    return cart_pred

def normalize_w(w_run, w_term):
    A = np.float64(np.array(list(w_run.items()))[:,1])
    B = np.float64(np.array(list(w_term.items()))[:,1])
    M = np.max([np.max(A), np.max(B)])
    if M == 0:
        return w_run, w_term
    for key, value in zip(w_run.keys(), w_run.values()):
        w_run[key] = value/M
    for key, value in zip(w_term.keys(), w_term.values()):
        w_term[key] = value/M
    return w_run, w_term

def vectorize_w(w_run, w_term, Keys_run, Keys_term):
    wv_run = np.zeros(len(w_run))
    wv_term = np.zeros(len(w_term))
    wv = np.zeros(len(w_run)+len(w_term))
    for i, k in enumerate(Keys_run):
        wv[i] = w_run[k]
        wv_run[i] = w_run[k]
    for i, k in enumerate(Keys_term):
        wv[i+len(w_run)] = w_term[k]
        wv_term[i] = w_term[k]
    return wv_run, wv_term, wv