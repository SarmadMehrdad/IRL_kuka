import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import crocoddyl
import mujoco
import mujoco.viewer
import mim_solvers
import numpy as np
from mim_robots.pybullet.env import BulletEnvWithGround
from mim_robots.robot_loader import load_bullet_wrapper, load_mujoco_model, get_robot_list, load_pinocchio_wrapper
from mim_robots.robot_list import MiM_Robots

# Robot simulator Mujoco
RobotInfo = MiM_Robots["iiwa"]
mj_model = load_mujoco_model("iiwa")
mj_data = mujoco.MjData(mj_model)
robot_simulator = load_pinocchio_wrapper("iiwa")
pin_model = robot_simulator.model
pin_collision_model = robot_simulator.collision_model
pin_visual_model = robot_simulator.visual_model
pin_data = pin_model.createData()
viz = MeshcatVisualizer(pin_model, pin_collision_model, pin_visual_model)
link_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]

# Robot simulator Mujoco
mj_model = load_mujoco_model("iiwa")
mj_data = mujoco.MjData(mj_model)

# Extract robot model
nq = robot_simulator.model.nq
nv = robot_simulator.model.nv
nu = nq; nx = nq+nv
q0 = np.array([0.2, 0.7, 0.3, -1.1, -0.3, 0.3, 0.])
# q0 = pin.neutral(pin_model)
v0 = np.zeros(nv)
idx = robot_simulator.index('A7')
pin.forwardKinematics(pin_model, pin_data, q0)
x0 = np.concatenate([q0, v0])
# print(pin_data.oMi[idx].translation)

# Add robot to Mujoco and initialize
mj_renderer = mujoco.Renderer(mj_model)
mujoco.mj_step(mj_model, mj_data)
mj_renderer.update_scene(mj_data)
mj_data.qpos = q0
mj_data.qvel = v0
mujoco.mj_forward(mj_model, mj_data)
mj_dt=1e-3