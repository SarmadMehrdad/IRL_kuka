import crocoddyl
import pinocchio as pin
from IPython.display import HTML
import mim_solvers
import numpy as np
import random
from matplotlib import animation
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds

'''
Args should have the following structure:

The 'type' will define the type of IRL algorithm to be used.
'type': {'mrinal', 'autoreg'}

The 'solver' will be the crocoddyl solver which contains the model and the cost function.

The 'w_run' and 'w_term' will be the initial weights for the running and terminal costs respectively.
We are not going to be relying on the solvers weights for the cost functions, but rather the weights passed in the args.

'dt' is the time step for the IRL algorithm.

'use_bad' is the flag that uses the bad trajectory samples provided in the args.
{True, False} -> 'xs_nopt' and 'us_nopt'

The 'normalize' param will indicate if the feature normalization should be used throughout the IRL iterations.
{True, False}

'N_samples' is the number of trajectory samples IRL will use to update the weights.

'K_set' is the number of trajectories used in the IRL iterations for the partition function.

'use_best' indicates if the best the trajectories with the closest cumulative feature counts to the optimal trajectory should be used.
{True, False}

'Lambda' is the weight regularization parameter for the IRL algorithm.

'KL_tol' is the tolerance for the KL divergence between the current and the previous trajectory sets.

'max_iter' is the maximum number of iterations for the IRL algorithm.

'min_iter' is the minimum number of iterations for the IRL algorithm.

'lr' is the learning rate for the IRL algorithm.
(0.0, 1.0)

'compare_desired' is the flag to compare the desired weights with the learned weights throughout the iterations.
If True, the desired weights should be passed in the args.
{True, False} -> 'des_run' and 'des_term'

'irl_iter' is the number of iterations for the IRL autoregressive loop.

'sqp_iter' is the number of iterations for the SQP solver for IRL autoregressive loop.

'next_traj' is the flag for determining the next trajectory to be used in the IRL autoregressive loop.
{'best, 'last', 'worst', 'optimal'}

'verbose' is the flag for the verbosity of the IRL algorithm.

'''


class IRL_Crocoddyl():
    def __init__(self,xs_opt, us_opt, args):
        self.xs_opt = np.stack(xs_opt)
        self.us_opt = np.stack(us_opt)
        self.args = args
        self.dt = args['dt']
        self.max_iter = args['max_iter']
        self.min_iter = args['min_iter']
        self.N_samples = args['N_samples']
        self.K_set = args['K_set']
        self.solver = args['solver']
        self.type = args['type']
        self.verbose = args['verbose']
        self.normalize = args['normalize']
        self.compare_desired = args['compare_desired']
        self.KL_tol = args['KL_tol']
        self.use_best = args['use_best']
        self.line_search_steps = args['line_search_steps']
        self.line_search_base = args['line_search_base']
        self.phi_opt = None
        self.cost_opt = None
        self.Xs = []
        self.Us = []
        self.phis = []
        self.costs = []
        self.phis_set = []
        self.costs_set = []
        self.cost_diffs = [np.inf]
        self.ws = []
        self.w_star = None
        self.Lambda = 0.0
        self.best_traj = None
        self.best_w = None
        self.last_traj = None
        self.last_w = None
        self.mu_phi = None
        self.steps = 0.5**(2*(np.array(list(range(self.line_search_steps)))))
        self.init_params()
 
    ########################################
    ########## Essential functions #########
    ########################################
    def init_params(self):
        # Solver related params
        self.T = self.solver.problem.T

        self.solver.termination_tolerance = 1e-4
        self.solver.with_callbacks = False


        self.nr_run = len(self.solver.problem.runningModels[0].differential.costs.costs.todict().keys())
        self.nr_term = len(self.solver.problem.terminalModel.differential.costs.costs.todict().keys())

        self.nr = self.nr_run + self.nr_term

        self.keys_run = list(self.solver.problem.runningModels[0].differential.costs.costs.todict().keys())
        self.keys_term = list(self.solver.problem.terminalModel.differential.costs.costs.todict().keys())
        if 'w_run' in self.args.keys() and 'w_term' in self.args.keys():
            self.w_run = self.args['w_run']
            self.w_term = self.args['w_term']
            self.prev_w_run = self.args['w_run']
            self.prev_w_term = self.args['w_term']
            self.ws.append(self.dict_to_vector(self.w_run, self.w_term))
        else:
            self.w_run, self.w_term = self.generate_zero_w()
            self.prev_w_run, self.prev_w_term = self.generate_zero_w()
            self.ws.append(self.dict_to_vector(self.w_run, self.w_term))

        self.update_solver_w(self.w_run, self.w_term)

        phi_opt, phis_opt = self.get_traj_features(self.xs_opt, self.us_opt)
        self.phis.append(phi_opt); self.phis_set.append(phis_opt)
        self.Xs.append(self.xs_opt); self.Us.append(self.us_opt)
        self.phi_opt = phi_opt; self.phis_opt = phis_opt

        if 'use_bad' in self.args:
            self.use_bad = self.args['use_bad']
            if self.use_bad:
                try:
                    self.xs_nopt = np.stack(self.args['xs_nopt'])
                    self.us_nopt = np.stack(self.args['us_nopt'])
                except:
                    print('Non optimal trajectory samples and weight vector not provided.')
                    self.use_bad = False
            else:
                self.xs_nopt, self.us_nopt = self.generate_bad_trajectory()

        phi_nopt, phis_nopt = self.get_traj_features(self.xs_nopt, self.us_nopt)
        self.phis.append(phi_nopt); self.phis_set.append(phis_nopt)
        self.Xs.append(self.xs_nopt); self.Us.append(self.us_nopt)
        self.opt_div = [1.0]

        self.d_phi = np.linalg.norm(phi_opt - phi_nopt)

        # IRL related params
        if 'Lambda' in self.args.keys():
            self.Lambda = self.args['Lambda']

        if self.compare_desired:
            try:
                self.w_star_run, self.w_star_term = self.args['des_run'], self.args['des_term']
                self.w_star = self.dict_to_vector(self.w_star_run, self.w_star_term)
            except:
                print('Desired weights not provided.')
                self.compare_desired = False
        
        if self.type == 'autoreg':
            try:
                self.irl_iter = self.args['irl_iter']
                self.sqp_iter = self.args['sqp_iter']
                self.next_traj = self.args['next_traj']
                self.lr = self.args['lr']
            except:
                print('IRL autoregressive loop parameters not provided. Please provide \'irl_iter\', \'next_traj\',  and \'sqp_iter\' parameters.') 
                print('Switching to Mrinal Algorithm')
                self.type = 'mrinal'

        return True

    
    def generate_bad_trajectory(self):
        w_run = self.w_run.copy()
        w_term = self.w_term.copy()
        self.update_solver_w(w_run, w_term)
        self.solver.termination_tolerance = 1e-4
        self.solver.with_callbacks = False
        xs_init = [self.xs_opt[i] for i in range(self.T+1)]
        us_init = [self.us_opt[i] for i in range(self.T)]
        self.solver.solve(xs_init, us_init, 100)
        return np.stack(self.solver.xs.copy()), np.stack(self.solver.us.copy())

    def get_traj_costs(self, phis, w_run, w_term):
        cost = 0.0
        cost_set = []
        
        # Terminal Featuressolver_args['test'] = curr_value
        phi = phis[-1]
        cost += np.sum(phi[self.nr_run:]*w_term)
        cost_set.append(cost)
        # Running Features
        for i in range(self.T-1,-1,-1):
            phi = phis[i]
            cost += np.sum(phi[:self.nr_run]*w_run)*self.dt
            cost_set.append(cost)

        return cost, cost_set
    
    def update_solver_w(self,w_run,w_term):
        for i in range(self.T):
            for key_ in self.keys_run: 
                self.solver.problem.runningModels[i].differential.costs.costs[key_].weight = w_run[key_]
        for key_ in self.keys_term: 
            self.solver.problem.terminalModel.differential.costs.costs[key_].weight = w_term[key_]
        
        return True

    
    def solve_autoreg(self, solver_args):

        def loop_termination_check(args):
            message = ''
            termination = False
            iteration = args['iteration']
            KL = args['KL_div']
            hard_terminate = args['hard_terminate']
            max_iter_termination = False
            KL_termination = False
            min_iter_termination = False
            if iteration >= self.max_iter:
                max_iter_termination = True
            if KL < self.KL_tol:
                KL_termination = True
            if iteration >= self.min_iter:
                min_iter_termination = True
            
            if min_iter_termination and KL_termination:
                termination = True
                message = 'KL divergence and minimum iteration termination criteria met.'
            elif max_iter_termination:
                termination = True
                message = 'Maximum iteration termination criteria met.'
            else:
                termination = False
            if iteration < 9:
                filler = ' '
            else: 
                filler = ''

            if 'KL_div_star' in args.keys():
                info = '||  {}{}  || {:.6f} || {:.6f} || {:.6f} || {:.6f} || {:.6f} || {} || {}'.format(
                    filler,
                    args['iteration']+1, 
                    args['KL_div'], 
                    args['KL_div_star'], 
                    args['Opt_Dev'], 
                    args['Cost_Diff'], 
                    args['Fcn_Val'], 
                    args['Step'],
                    args['Step_Type'])
            
            if hard_terminate and min_iter_termination:
                termination = True
                message = 'Hard termination criteria met.'
            return termination, info, message

        @staticmethod
        def fcn_w(x, opt_ind, nopt_ind):
            output = 0.0
            w_run = x[:self.nr_run]; w_term = x[self.nr_run:]
            ll = log_likelihood(opt_ind, nopt_ind, w_run, w_term)
            output = -ll
            output += self.Lambda*np.linalg.norm(x)
            return output

        def check_vector_values(V):
            inds = np.zeros_like(V)
            for i, v in enumerate(V):
                if v <= 0:
                    inds[i] = 0
                else:
                    inds[i] = 1/v
            return np.argmax(inds)

        def get_inds():
            inds = []
            if len(self.Xs) > self.K_set:
                if self.use_best:
                    # vals = np.array(self.opt_div[1:])
                    vals = np.array(self.cost_diffs)
                    inds = list((np.argsort(vals)+1)[:self.K_set])
                else:
                    inds = list(range(len(self.Xs)-self.K_set,len(self.Xs)))
            else:
                inds = list(range(len(self.Xs)))
                inds = inds[1:]
            return inds

        def initiate_trajectory():
            if self.next_traj == 'best':
                X, U = self.get_best_trajectory()
            elif self.next_traj == 'last':
                X = [self.Xs[-1][i] for i in range(self.T+1)]
                U = [self.Us[-1][i] for i in range(self.T)]
            elif self.next_traj == 'worst':
                # X, U = self.get_worst_trajectory()
                X = [self.Xs[1][i] for i in range(self.T+1)]
                U = [self.Us[1][i] for i in range(self.T)]
            elif self.next_traj == 'optimal':
                X = [self.xs_opt[i] for i in range(self.T+1)]
                U = [self.us_opt[i] for i in range(self.T)]
            return X, U

        def log_likelihood(opt_ind, nopt_ind, w_run, w_term):
            ll = 0.0
            phis_opt = self.phis_set[opt_ind]
            inds = [opt_ind]; inds.extend(nopt_ind); phis_set = [self.phis_set[i] for i in inds]
            costs_set = []
            if self.normalize:
                phis_set = self.normalize_features(phis_set)
                for phis in phis_set:
                    _, costs = self.get_traj_costs(phis, w_run, w_term)
                    costs_set.append(costs)
            else:
                for phis in phis_set:
                    _, costs = self.get_traj_costs(phis, w_run, w_term)
                    costs_set.append(costs)
            
            samples = np.linspace(0,self.T-1,self.N_samples).astype(int)

            norm_phi = np.zeros(shape=(self.T, self.K_set, self.nr))
            for i in range(self.T):
                for j, phi in enumerate(phis_set[1:]):
                    norm_phi[i,j] = phi[i] - phis_opt[i]
            norm_phi = np.linalg.norm(norm_phi, axis = 2)
            max_d_phi = np.max(norm_phi, axis=1)
            sum_phi = np.sum(norm_phi, axis = 1) + 1
            mu_phi = np.mean(norm_phi, axis = 1)
            for i in samples:
                c_o = costs_set[0][i]
                p_o = phis_opt[i]
                # num = 1.0
                num = 1.0/sum_phi[i]
                den = num
                for j, c_no in enumerate(costs_set[1:]):
                    val =  (norm_phi[i,j]/sum_phi[i])*(np.exp(-(c_no[i] - c_o))) # Working !!
                    # val =  norm_phi[i,j]*(np.exp(-(c_no[i] - c_o))) # Working !!
                    # val =  np.exp(norm_phi[i,j]/sum_phi[i])*(np.exp(-(c_no[i] - c_o))) 
                    # val =  np.exp(norm_phi[i,j]/max_d_phi[i])*(np.exp(-(c_no[i] - c_o)))
                    # val =  np.exp(-(c_no[i] - c_o)*1e-1)
                    # val =  np.exp(-(c_no[i] - c_o))
                    den += np.max([0.0, np.min([1e+300, val])])
                # ll += np.log(num/den)
                # ll += mu_phi[i]*np.log(num/den)
                # ll += (1 - i/self.T)*np.log(num/den)
                ll += (1 - i/self.T)*mu_phi[i]*np.log(num/den)
            self.mu_phi = mu_phi
            return ll
        
        def line_search(w_prev, dw, solver_args, type = 'cost'):
            solver_args['hard_terminate'] = False
            w_temp = w_prev.copy()
            chosen_step = False
            step_ind = 0
            cost_diffs = []
            opt_diffs = []
            if type == 'none':
                w_loop = w_prev + dw
                chosen_step = True
                solver_args['Step'] = 1.0
            while not chosen_step:
                step = self.steps[step_ind]
                w_curr = w_temp + step*dw
                w_curr = self.normalize_vector(w_curr)
                w_run_temp, w_term_temp = self.vector_to_dict(w_curr[:self.nr_run], w_curr[self.nr_run:])
                self.update_solver_w(w_run_temp, w_term_temp)
                xs_init, us_init = initiate_trajectory()
                self.solver.solve(xs_init, us_init, self.sqp_iter)
                phi, phi_set = self.get_new_traj_features()
                if type in ['opt', 'both']:
                    opt_div = (np.linalg.norm(phi - self.phis[0]))/self.d_phi
                    condition = opt_div < self.opt_div[-1]
                    if type == 'both':
                        condition_opt = condition
                if type in ['cost', 'both']:
                    w_run_temp_v, w_term_temp_v, _ = self.dict_to_vector(w_run_temp, w_term_temp)
                    c, _ = self.get_traj_costs(phi_set, w_run_temp_v, w_term_temp_v)
                    c_opt, _ = self.get_traj_costs(self.phis_set[0], w_run_temp_v, w_term_temp_v)
                    condition = (c - c_opt)**2 < self.cost_diffs[-1]
                    if type == 'both':
                        condition_cost = condition

                if type != 'both':
                    if condition:
                        chosen_step = True
                        solver_args['Step'] = step_ind
                        solver_args['Step_Type'] = type
                        w_loop = w_temp + step*dw
                    else:
                        chosen_step = False
                        step_ind += 1
                        if step_ind == len(self.steps):
                            solver_args['Step'] = 'Stop'
                            chosen_step = True
                            solver_args['hard_terminate'] = True
                            w_loop = w_prev
                            break
                else:
                    if condition_cost:
                        chosen_step = True
                        solver_args['Step'] = step_ind
                        solver_args['Step_Type'] = 'Cost'
                        w_loop = w_temp + step*dw
                    else:
                        chosen_step = False
                        step_ind += 1
                        opt_diffs.append(self.opt_div[-1] - opt_div)
                        if step_ind == len(self.steps):
                            if opt_diffs[np.argmax(np.stack(opt_diffs))] > 0:
                                ind = check_vector_values(np.array(opt_diffs))
                                step = self.steps[ind]
                                chosen_step = True
                                solver_args['Step'] = ind
                                solver_args['Step_Type'] = 'Cost based on opt'
                                w_loop = w_temp + step*dw
                            else:
                                solver_args['Step'] = 0
                                chosen_step = True
                                solver_args['hard_terminate'] = True
                                w_loop = w_prev
                                break
            
            return w_loop, solver_args
                
        def solve(solver_args):
            terminate = False
            _, _, w_loop = self.dict_to_vector(self.w_run, self.w_term)
            lb = 0; ub = np.inf
            # lb = 0; ub = 1.0
            bnds = Bounds(lb, ub)
            tol = 1e-10
            opt_ind = 0
            opt_div = 100.0
            nopt_inds = get_inds()
            options = {'maxiter': self.irl_iter, 'iprint': -1,'ftol': 1e-10 ,'gtol' : 1e-10}
            print('-- iter -- KL_div ---- KL_des ---- Opt Div --- Cost Diff --- Fcn Val ---- Step ---')
            while not terminate:
                prev_opt_div = opt_div
                w_prev = w_loop.copy()
                res = minimize(fcn_w, 
                            w_loop, 
                            args=(opt_ind, nopt_inds), 
                            bounds=bnds, 
                            method='L-BFGS-B', 
                            tol = tol,
                            options=options)
                
                w_curr = res.x.copy()
                dw = (self.normalize_vector(np.squeeze(w_curr)) - w_prev)*self.lr

                w_loop, solver_args = line_search(w_prev, dw, solver_args, type = self.line_search_base)

                if not solver_args['hard_terminate']:
                    w_loop = self.normalize_vector(w_loop)
                    w_run, w_term = self.vector_to_dict(w_loop[:self.nr_run], w_loop[self.nr_run:])
                    self.w_run = w_run.copy(); self.w_term = w_term.copy()
                    self.update_solver_w(w_run, w_term)
                    xs_init, us_init = initiate_trajectory()
                    self.solver.solve(xs_init, us_init, self.sqp_iter)
                    new_x = np.stack(self.solver.xs.copy())
                    new_u = np.stack(self.solver.us.copy())
                    self.Xs.append(new_x.copy())
                    self.Us.append(new_u.copy())
                    phi, phi_set = self.get_new_traj_features()
                    self.phis.append(phi); self.phis_set.append(phi_set)
                    w_run_v, w_term_v, _ = self.dict_to_vector(w_run, w_term)
                    c, _ = self.get_traj_costs(phi_set, w_run_v, w_term_v)
                    c_nopt, _ = self.get_traj_costs(self.phis_set[1], w_run_v, w_term_v)
                    c_opt, _ = self.get_traj_costs(self.phis_set[0], w_run_v, w_term_v)
                    # c_nopt, _ = self.get_traj_costs(self.phis_set[1], w_run_v, w_term_v)
                    # solver_args['Cost_Diff'] = ((c - c_opt)/(c_opt - c_nopt))**2; self.cost_diffs.append(solver_args['Cost_Diff'])
                    solver_args['Cost_Diff'] = (c - c_opt)**2; self.cost_diffs.append(solver_args['Cost_Diff'])
                    solver_args['Opt_Dev'] = (np.linalg.norm(phi - self.phi_opt))/self.d_phi; self.opt_div.append(solver_args['Opt_Dev'])
                    solver_args['Fcn_Val'] = fcn_w(w_loop, 0, nopt_inds)
                    self.ws.append(self.dict_to_vector(self.w_run, self.w_term))
                
                    nopt_inds = get_inds()
                    opt_ind = 0                
                    
                    solver_args['KL_div'] = self.KL_divergence(log_likelihood, [opt_ind] + nopt_inds, w_prev, w_loop)
                    if self.compare_desired:
                        solver_args['KL_div_star'] = self.KL_divergence(log_likelihood, [opt_ind] + nopt_inds, self.w_star[-1], w_loop)
                terminate, info, message = loop_termination_check(solver_args)
                if self.verbose and not terminate:
                    print(info)
                solver_args['iteration'] += 1
            # if not self.verbose:
            print(message)

        
        solve(solver_args)

        return True
    
    def get_new_traj_features(self):
        Phi = np.zeros((self.nr))
        Phi_set = []

        # Terminal Features
        for j, k in enumerate(self.keys_term):
            Phi[j + self.nr_run] += self.solver.problem.terminalData.differential.costs.costs[k].cost
        Phi_set.append(Phi.copy())
        
        # Running Features
        for i in range(self.T-1,-1,-1):
            for j, k in enumerate(self.keys_run):
                Phi[j] += self.solver.problem.runningDatas[i].differential.costs.costs[k].cost
            Phi_set.append(Phi.copy())
        
        return Phi, Phi_set
    
    def get_traj_features(self, xs, us):
        Phi = np.zeros((self.nr))
        Phi_set = []

        # Terminal Features
        X = xs[-1]
        for j, k in enumerate(self.keys_term):
            self.solver.problem.terminalModel.differential.costs.costs[k].cost.calc(self.solver.problem.terminalData.differential.costs.costs[k], X)
            self.solver.problem.terminalModel.differential.costs.costs[k].cost.calcDiff(self.solver.problem.terminalData.differential.costs.costs[k], X)
            Phi[j + self.nr_run] += self.solver.problem.terminalData.differential.costs.costs[k].cost
        Phi_set.append(Phi.copy())
        
        # Running Features
        for i in range(self.T-1,-1,-1):
            X = xs[i]; U = us[i]
            self.solver.problem.runningModels[i].differential.actuation.calc(self.solver.problem.runningDatas[i].differential.multibody.actuation, X, U)
            self.solver.problem.runningModels[i].differential.actuation.calcDiff(self.solver.problem.runningDatas[i].differential.multibody.actuation, X, U)
            for j, k in enumerate(self.keys_run):
                self.solver.problem.runningModels[i].differential.costs.costs[k].cost.calc(self.solver.problem.runningDatas[i].differential.costs.costs[k], X, U)
                self.solver.problem.runningModels[i].differential.costs.costs[k].cost.calcDiff(self.solver.problem.runningDatas[i].differential.costs.costs[k], X, U)
                Phi[j] += self.solver.problem.runningDatas[i].differential.costs.costs[k].cost
            Phi_set.append(Phi.copy())
        
        return Phi, Phi_set
    
    def solve_irl(self):
        if self.type == 'mrinal':
            self.solve_mrinal()
        elif self.type == 'autoreg':
            solver_args = {
                'hard_terminate': False,
                'iteration': 0,
                'KL_div': 1.0,
                'min_iter': self.min_iter,
                'max_iter': self.max_iter,
                'KL_tol': self.KL_tol,
                'KL_div_star': None,
                'Opt_Dev': None,
                'Cost Diff': None,
                'Fcn_Val': 1.0,
                'Step': 1.0,
                'Step_Type': 'None'
            }
            self.solve_autoreg(solver_args)
            self.report_results()
        else:
            print('IRL type not recognized.')

    
    
    ########################################
    ############ Handy functions ###########
    ########################################
    def dict_to_vector(self, w_run, w_term):
        wv_run = np.zeros(len(w_run))
        wv_term = np.zeros(len(w_term))
        wv = np.zeros(len(w_run)+len(w_term))
        for i, k in enumerate(self.keys_run):
            wv[i] = w_run[k]
            wv_run[i] = w_run[k]
        for i, k in enumerate(self.keys_term):
            wv[i+len(w_run)] = w_term[k]
            wv_term[i] = w_term[k]
        return wv_run, wv_term, wv
    
    def vector_to_dict(self, wv_run, wv_term):
        w_run = {}
        w_term = {}
        for i, k in enumerate(self.keys_run):
            w_run[k] = wv_run[i]
        for i, k in enumerate(self.keys_term):
            w_term[k] = wv_term[i]
        return w_run, w_term
    
    def normalize_w_dict(self,w_run,w_term):
        A = np.float64(np.array(list(w_run.items()))[:,1])
        B = np.float64(np.array(list(w_term.items()))[:,1])
        M = np.max([np.max(A), np.max(B)])
        for key, value in zip(w_run.keys(), w_run.values()):
            w_run[key] = value/M
        for key, value in zip(w_term.keys(), w_term.values()):
            w_term[key] = value/M
        return w_run, w_term
    
    def normalize_w_vector(self,w_run,w_term):
        A = np.float64(w_run)
        B = np.float64(w_term)
        M = np.max([np.max(A), np.max(B)])
        w_run = w_run/M
        w_term = w_term/M
        return w_run, w_term

    def normalize_vector(self, wv):
        M = np.max(np.abs(wv))
        if M == 0:
            return wv
        return wv/M
    
    def normalize_features(self, phis_set):
        normalized_phis = phis_set.copy()
        if len(phis_set) == 1:
            return phis_set
        else:
            for i in range(len(phis_set[0])):
                phi_ = []
                for phi in phis_set:
                    phi_.append(phi[i])
                phi_ = np.stack(phi_)
                std_phi = np.std(phi_, axis = 0)
                std_phi[std_phi == 0] = 1.0
                for j in range(len(normalized_phis)):
                    normalized_phis[j][i] /= std_phi
        return normalized_phis
    
    def generate_zero_w(self):
        w_run = {}
        w_term = {}
        for k in self.keys_run:
            w_run[k] = 0.0
        for k in self.keys_term:
            w_term[k] = 0.0
        return w_run, w_term
    
    def get_best_trajectory(self):
        # opt_div = np.array(self.opt_div)
        # best_ind = np.argmin(opt_div)
        cost_diffs = np.array(self.cost_diffs)
        best_ind = np.argmin(cost_diffs)
        X = [self.Xs[best_ind][i] for i in range(self.T+1)]
        U = [self.Us[best_ind][i] for i in range(self.T)]
        return X, U

    def get_worst_trajectory(self):
        opt_div = np.array(self.opt_div)
        worst_ind = np.argmax(opt_div)
        X = [self.Xs[worst_ind][i] for i in range(self.T+1)]
        U = [self.Us[worst_ind][i] for i in range(self.T)]
        return X, U

    def KL_divergence(self, likelihood_fcn, inds, w1, w2):
        KL = 0.0
        Px = np.zeros(len(inds))
        Qx = np.zeros(len(inds))
        for i in range(len(inds)):
            temp_inds = inds.copy(); temp_inds.pop(i)
            w_run_1 = w1[:self.nr_run]; w_term_1 = w1[self.nr_run:]
            w_run_2 = w2[:self.nr_run]; w_term_2 = w2[self.nr_run:]
            Px[i] = likelihood_fcn(inds[i], temp_inds, w_run_1, w_term_1)
            Qx[i] = likelihood_fcn(inds[i], temp_inds, w_run_2, w_term_2)
        Px = Px/np.sum(Px)
        Qx = Qx/np.sum(Qx)
        kl_array = Px*np.log(Px/Qx)
        KL = np.sum(kl_array)
        return KL

    def print_info(self):
        print('IRL Parameters:')
        print('Initial Running Weight: ', self.w_run)
        print('Initial Terminal Weight: ', self.w_term)
        print('Type: ', self.type)
        print('Set Size: ', self.K_set)
        print('Sample Size: ' , self.N_samples)
        print('Lambda: ', self.Lambda)
        print('SQP Iterations: ', self.sqp_iter)
        print('IRL Max Iteration: ', self.max_iter)
        print('Sample Time: ', self.dt)

    def report_results(self):

        self.last_traj = [self.Xs[-1], self.Us[-1]]
        self.last_w = self.ws[-1]
        self.last_ind = len(self.Xs)-1

        best_ind = np.argmin(np.stack(self.opt_div[1:]))
        self.best_traj = [self.Xs[best_ind+1], self.Us[best_ind+1]]
        self.best_w = self.ws[best_ind]
        self.best_ind = best_ind