import pinocchio as pin
import crocoddyl
import numpy as np
import hppfcl
from Kuka_utils import *

def init_robot(robot_simulator, q0, v0, obs_set, ee_trans, w_run, w_term, dt, T):
    nq = robot_simulator.model.nq
    nv = robot_simulator.model.nv
    nu = nq; nx = nq+nv
    pin_model = robot_simulator.model
    # pin_collision_model = robot_simulator.collision_model
    # pin_visual_model = robot_simulator.visual_model
    pin_data = pin_model.createData()
    link_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    x0 = np.concatenate([q0, v0])

    # Create a collision capsules for Kuka
    capsul_disp = []
    for i,ln in enumerate(link_names[:-1]):
        capsul_disp.append(robot_simulator.placement(q0,robot_simulator.index(ln)+1).translation - robot_simulator.placement(q0,robot_simulator.index(ln)).translation)
    capsul_disp = np.stack(capsul_disp)
    capsul_lengths = np.linalg.norm(capsul_disp,axis=-1)
    capsul_radius = 0.04
    capsul_tol = capsul_radius/4
    ig_link_names = []
    pin_joint_ids = []
    geomModel           = pin.GeometryModel() 
    for i,ln in enumerate(link_names[:-1]):
        pin_link_id         = pin_model.getFrameId(ln)
        pin_joint_id        = pin_model.getJointId(ln)
        pin_joint_ids.append(pin_joint_id)
        placement = pin.SE3.Identity()
        placement.rotation = robot_simulator.placement(q0,pin_joint_id).rotation.T
        placement.translation += np.array([0, 0, capsul_lengths[i]/2])@placement.rotation.T
        # placement.translation += (capsul_disp[i]/2)@placement.rotation.T
        if ln == 'A6':
            placement.translation += np.array([0.03, 0, 6.0700e-02])
        ig_link_names.append(geomModel.addGeometryObject(pin.GeometryObject("arm_link_"+str(i+1), 
                                                        pin_model.joints[pin_model.getJointId(ln)].id,
                                                        placement,
                                                        hppfcl.Capsule(capsul_radius, capsul_lengths[i]/2 - capsul_tol))))
    
    # Create obstacles in the world
    # obs_num = obs_set.obs_num
    for i,(p,l) in enumerate(zip(obs_set.obs_p, obs_set.obs_l)):
        obsObj = create_obs(pin_model, p, pin.SE3.Identity().rotation, pin_model, [l]*3, i)
        ig_obs = geomModel.addGeometryObject(obsObj)
        for j in ig_link_names[-1:]:
            # geomModel.addCollisionPair(pin.CollisionPair(ig_obs, ig_link_names[j])) # Mine
            geomModel.addCollisionPair(pin.CollisionPair(ig_link_names[j],ig_obs)) # Original

    
    # endeff frame translation goal
    endeff_frame_id = pin_model.getFrameId("contact")
    endeff_joint_id = pin_model.getJointId("contact")
    # endeff_translation = pin_data.oMf[endeff_frame_id].translation.copy()
    endeff_translation = ee_trans

    ######################## Costs ########################

    state = crocoddyl.StateMultibody(pin_model)
    actuation = crocoddyl.ActuationModelFull(state)
    runningCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel = crocoddyl.CostModelSum(state)

    # Data collectors
    dataCollectorAct = crocoddyl.DataCollectorActMultibody(pin_data, crocoddyl.ActuationDataAbstract(actuation))
    dataCollectorState = crocoddyl.DataCollectorMultibody(pin_data)
    dataCollectorTranslation = crocoddyl.DataCollectorMultibody(pin_data)
    dataCollectorCollision = crocoddyl.DataCollectorMultibody(pin_data)
    # Control regularization cost
    uResidual = crocoddyl.ResidualModelControlGrav(state)
    uRegCost = crocoddyl.CostModelResidual(state,uResidual)
    uCostData = uRegCost.createData(dataCollectorAct)
    # uCostData 
    # State regularization cost
    xResidual = crocoddyl.ResidualModelState(state, x0)
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    xCostData = xRegCost.createData(dataCollectorState)
    # endeff frame translation cost
    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, endeff_frame_id, endeff_translation)
    frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
    translationCostData = frameTranslationCost.createData(dataCollectorTranslation)

    runningCostData = {}
    terminalCostData = {}
    collisionPairCostData = []
    residualCollision = []
    costCollision = []

    runningCostModel.addCost("xReg", xRegCost, w_run["xReg"])
    runningCostModel.addCost("uReg", uRegCost, w_run["uReg"])
    runningCostModel.addCost("translation", frameTranslationCost, w_run["translation"])
    terminalCostModel.addCost("xReg", xRegCost, w_term["xReg"])
    terminalCostModel.addCost("translation", frameTranslationCost, w_term["translation"])

    # Add collision cost
    c = 0
    
    for i in range(len(geomModel.collisionPairs)):
        #  Collision Cost
        activationCollision = crocoddyl.ActivationModel2NormBarrier(3, obs_set.col_r[i])
        residualCollision.append(crocoddyl.ResidualModelPairCollision(state, nu, geomModel, i, pin_joint_ids[-1]))
        costCollision.append(crocoddyl.CostModelResidual(state, activationCollision, residualCollision[i]))
        collisionPairCostData.append(costCollision[i].createData(dataCollectorCollision))
        runningCostModel.addCost("collision"+str(c), costCollision[i], w_run["collision"+str(c)])
        terminalCostModel.addCost("collision"+str(c), costCollision[i], w_term["collision"+str(c)])
        runningCostData['collision'+str(c)] = costCollision[i].createData(dataCollectorCollision)
        terminalCostData['collision'+str(c)] = costCollision[i].createData(dataCollectorCollision)
        c += 1

    runningCostData['xReg'] = xCostData
    runningCostData['uReg'] = uCostData
    runningCostData['translation'] = translationCostData
    terminalCostData['xReg'] = xCostData
    terminalCostData['translation'] = translationCostData

    running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
    terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel)

    # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
    runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)
    # runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
    # terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
    solver = mim_solvers.SolverSQP(problem)

    return solver

def update_solver_weights(solver, T, w_run, w_term):
    for i in range(T):
        for key_ in solver.problem.runningModels[i].differential.costs.costs.todict(): 
            solver.problem.runningModels[i].differential.costs.costs[key_].weight = w_run[key_]
    for key_ in solver.problem.terminalModel.differential.costs.costs.todict(): 
        solver.problem.terminalModel.differential.costs.costs[key_].weight = w_term[key_]

def get_traj_feature_and_cost(solver, xs, us, dt):
    T = len(xs)-1
    cost = 0.0
    Keys_run = list(solver.problem.runningModels[0].differential.costs.costs.todict().keys())
    Keys_term = list(solver.problem.terminalModel.differential.costs.costs.todict().keys())
    nr_run = len(Keys_run); nr_term = len(Keys_term); nr = nr_run + nr_term
    Phi = np.zeros((nr))
    for i in range(T):
        for j, k in enumerate(Keys_run):
            d = solver.problem.runningDatas[i].differential.costs.costs[k].copy()
            w = solver.problem.runningModels[i].differential.costs.costs[k].weight
            solver.problem.runningModels[i].differential.costs.costs[k].cost.calc(d, xs[i], us[i])
            solver.problem.runningModels[i].differential.costs.costs[k].cost.calcDiff(d, xs[i], us[i])
            Phi[j] += d.cost
            cost += d.cost*w*dt
    for j, k in enumerate(Keys_term):
        d = solver.problem.terminalData.differential.costs.costs[k].copy()
        w = solver.problem.terminalModel.differential.costs.costs[k].weight
        solver.problem.terminalModel.differential.costs.costs[k].cost.calc(d, xs[-1])
        solver.problem.terminalModel.differential.costs.costs[k].cost.calcDiff(d, xs[-1])
        Phi[j + nr_run] += d.cost
        cost += d.cost*w
            
    return Phi, cost

def get_traj_feature_and_cost_multiple(solver, xs, us, dt, num = None):
    T = len(xs)
    cost = 0.0
    if num == 1 or num is None:
        intervals = [0]
    else:
        intervals = np.linspace(0,T,num).astype(int)
    ind = intervals[-1]
    cost_set = []
    Keys_run = list(solver.problem.runningModels[0].differential.costs.costs.todict().keys())
    Keys_term = list(solver.problem.terminalModel.differential.costs.costs.todict().keys())
    nr_run = len(Keys_run); nr_term = len(Keys_term); nr = nr_run + nr_term
    Phi = np.zeros((nr))
    Phi_set = []
    # Terminal Features
    for j, k in enumerate(Keys_term):
        d = solver.problem.terminalData.differential.costs.costs[k].copy()
        w = solver.problem.terminalModel.differential.costs.costs[k].weight
        solver.problem.terminalModel.differential.costs.costs[k].cost.calc(d, xs[-1])
        solver.problem.terminalModel.differential.costs.costs[k].cost.calcDiff(d, xs[-1])
        Phi[j + nr_run] += d.cost
        cost += d.cost*w
    if ind == T:
        Phi_set.append(Phi.copy())
        cost_set.append(cost)
        intervals = intervals[:-1]
    # Running Features
    for i in range(T-2,-1,-1):
        ind = intervals[-1]
        for j, k in enumerate(Keys_run):
            d = solver.problem.runningDatas[i].differential.costs.costs[k].copy()
            w = solver.problem.runningModels[i].differential.costs.costs[k].weight
            solver.problem.runningModels[i].differential.costs.costs[k].cost.calc(d, xs[i], us[i])
            solver.problem.runningModels[i].differential.costs.costs[k].cost.calcDiff(d, xs[i], us[i])
            Phi[j] += d.cost
            cost += d.cost*w*dt
        if ind == i:
            Phi_set.append(Phi.copy())
            cost_set.append(cost)
            intervals = intervals[:-1]
            
    return Phi, cost, Phi_set[::-1], cost_set[::-1]