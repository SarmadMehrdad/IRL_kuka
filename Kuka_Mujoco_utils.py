import mujoco
import mujoco.viewer
import pinocchio as pin
import numpy as np
import time

def init_mujoco(mj_model, mj_data, q0, v0, obs_set, ee_trans):
    mj_data.qpos = q0
    mj_data.qvel = v0
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data, show_left_ui=False, show_right_ui=False)
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
    viewer.user_scn.ngeom = obs_set.obs_num + 1
    for i, (p,l) in enumerate(zip(obs_set.obs_p, obs_set.obs_l)):
        mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[l, l, l],
                pos=p,
                mat=np.eye(3).flatten(),
                rgba=np.array([1.,0.,0.,.5])
            )
    
    mujoco.mjv_initGeom(
              viewer.user_scn.geoms[viewer.user_scn.ngeom-1],
              type=mujoco.mjtGeom.mjGEOM_SPHERE,
              size=[0.05, 0.05, 0.05],
              pos=ee_trans,
              mat=np.eye(3).flatten(),
              rgba=np.array([0.,1.,0.,.5])
          )
    viewer.sync()
    return viewer

def run_traj(viewer, mj_model, mj_data, pin_model, pin_data,  x, u, dt):
    nq = pin_model.nq
    for x_, u_ in zip(x[:-1], u): 
    
        # Send torque to simulator & step simulator
        # Mujoco Environment Update
        mj_data.ctrl = u_
        mj_data.qpos = x_[:nq]
        mj_data.qvel = x_[nq:]
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()
        
        # Measure new state from Mujoco
        q_ = mj_data.qpos
        v_ = mj_data.qvel

        # Update pinocchio model
        pin.forwardKinematics(pin_model, pin_data, q_, v_)
        pin.computeJointJacobians(pin_model, pin_data, q_)
        pin.framesForwardKinematics(pin_model, pin_data, q_)
        pin.updateFramePlacements(pin_model, pin_data)  
        # pin.computeKineticEnergy(pin_model, pin_data, q_, v_)
        # pin.computePotentialEnergy(pin_model, pin_data, q_)
        time.sleep(dt)
        
    x_ = x[-1]
    mj_data.qpos = x_[:nq]
    mj_data.qvel = x_[nq:]
    mujoco.mj_step(mj_model, mj_data)
    viewer.sync()