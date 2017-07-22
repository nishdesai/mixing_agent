from frozen_lake import *
import numpy as np, numpy.random as nr, gym
import matplotlib.pyplot as plt
from gym.spaces import prng

class MDP(object):
    def __init__(self, env):
        P, nS, nA, desc = MDP.env2mdp(env)
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means (used for plotting)
        self.env = env
        self.T = self.get_transition_matrix()

    def env2mdp(env):
        return {s : {a : [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, env.nS, env.nA, env.desc
    
    def get_transition_matrix(self):
        """Return a matrix with index S,A,S' -> P(S'|S,A)"""
        T = np.zeros([self.nS, self.nA, self.nS])
        for s in range(self.nS):
            for a in range(self.nA):
                transitions = self.P[s][a]
                s_a_s = {t[1]:t[0] for t in transitions}
                for s_prime in range(self.nS):
                    if s_prime in s_a_s:
                        T[s, a, s_prime] = s_a_s[s_prime]
        return T


def softmax(x1,x2):
    max_x = np.amax((x1,x2))
    min_x = np.amin((x1,x2))
    return max_x + np.log(1+np.exp(min_x - max_x))

def compute_value_boltzmann(mdp, gamma, reward, horizon = None, threshold=1e-4):
    """
    Find the optimal value function via value iteration with the max-ent Bellman backup 
    given at https://graphics.stanford.edu/projects/gpirl/gpirl_supplement.pdf.

    reward: Vector of rewards for each state.
    threshold: Convergence threshold.
    gamma: MDP gamma factor. float.
    -> Array of values for each state
    """
    
    if horizon is not None:
        gamma = 1
    
    v = np.zeros(mdp.nS)
    
    # This is how it is supposed to be; running into numerical problems for some reason
    #v = reward

    t = 0
    diff = float("inf")
    while diff > threshold:
        v_prev = np.copy(v)
        diff = 0
        for s in range(mdp.nS):
            v_s_new = 0
            for a in range(mdp.nA):
                if a == 0:
                    v_s_new += reward[s] + gamma * np.dot(mdp.T[s, a, :], v_prev)  
                else:
                    v_s_new = softmax(v_s_new, reward[s] + gamma * np.dot(mdp.T[s, a, :], v_prev))
            
                if np.sum(np.isnan(v_s_new)) > 0: 
                    raise Exception('NaN encountered at iteration ', t, 'state',s, ' action ', a)
            
            v[s] = v_s_new

            
        new_diff = np.amax(abs(v_prev - v))
        if new_diff > diff: diff = new_diff
        
        t+=1
        if horizon is not None:
            if t==horizon: break
    
    return v


def compute_policy(mdp, gamma, reward=None, V=None, horizon=None, threshold=1e-4):
    """
    Computes the Boltzmann policy from the value function
    
    -> Array of shape (mdp.nS, mdp.nA), each value p[s,a] is the probability of taking action a in state s
    """

    if reward is None and V is None: raise Exception('Cannot compute V: no reward provided')
    if V is None: V = compute_value_boltzmann(mdp, gamma, reward, horizon=horizon, threshold=threshold)
    #print(V)
    
    policy = np.zeros((mdp.nS, mdp.nA))
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            policy[s,a] = np.exp(reward[s] - V[s] + np.dot(mdp.T[s, a,:], gamma * V))
    

    # Hack for finite horizon length to make the probabilities sum to 1:
    policy = policy / np.sum(policy, axis=1).reshape((mdp.nS, 1))

    if np.sum(np.isnan(policy)) > 0: raise Exception('NaN encountered in policy')
    
    return policy


def generate_trajectories(mdp, policy, T=20, num_traj=50):
    s = mdp.env.reset()
    
    trajectories = np.zeros([num_traj, T, 2]).astype(int)
    
    for d in range(num_traj):
        for t in range(T):
            action = np.random.choice(range(mdp.nA), p=policy[s, :])
            trajectories[d, t, :] = [s, action]
            s, _, _, _ = mdp.env.step(action)
        s = mdp.env.reset()
    
    return trajectories


def compute_irl_log_likelihood(mdp, gamma, trajectories, V, r):

    L_D = 0

    for traj in trajectories:
        for (s, a) in traj:
            # This is Q[s,a] - V[s]
            L_D += r[s] + np.dot(mdp.T[s,a,:], gamma * V) - V[s]
    
    return L_D


def compute_s_a_visitations(mdp, gamma, trajectories):
    """
    Computes the empirical state and state-action visitation frequencies from 
    the expert trajectories
    """
    
    mu_hat_sa = np.zeros((mdp.nS, mdp.nA))
    init_s = np.zeros((mdp.nS))
    for traj in trajectories:
        for (s, a) in traj:
            mu_hat_sa[s, a] += 1
            init_s[s] += 1

            init_s -= gamma * mdp.T[s,a,:]
            # Same as the line above but slower:
            #for (s_prime, p_transition) in enumerate(t1[s,a,:]):
            #    init_s[s_prime] -= gamma * p_transition
    
    init_s = init_s / (trajectories.shape[0] * trajectories.shape[1])
    mu_hat_sa = mu_hat_sa / (trajectories.shape[0] * trajectories.shape[1])
    if np.sum(np.isnan(mu_hat_sa)) > 0: raise Exception('NaN encountered')
    
    return(mu_hat_sa, init_s)


def compute_D(mdp, gamma, V, policy, init_s, horizon=None, D = None, threshold = 1e-6):
    """
    Computes occupancy measure of a MDP under a given policy -- 
    the expected discounted number of times that policy π visits state s.
    """
    assert V.shape[0] == mdp.nS
    assert policy.shape == (mdp.nS, mdp.nA)   
    
    assert np.sum(np.isnan(V)) == 0
    assert np.sum(np.isnan(policy)) == 0
    assert np.sum(np.isnan(init_s)) == 0
        
    
    if D is None: D = init_s    
    else: D = np.tile(D, (mdp.nA, 1))
    
    
    if horizon is not None:
        gamma = 1
        t = 1
    
    diff = float("inf")
    while diff > threshold:
        D_new = np.zeros_like(D)
        
        for s in range(mdp.nS):
            for s_prime in range(mdp.nS):
                for a_prime in range(mdp.nA):
                    D_new[s_prime] += (policy[s, a_prime] 
                                              * (gamma * mdp.T[s,a_prime,s_prime] * D[s]))

            if np.sum(D_new>1e4) > 0: print('D_new > 1e04, iteration', t)
        
        diff = np.amax(abs(D - D_new))    
        D = np.copy(D_new)
        
        if horizon is not None:
            t+=1
            if t==horizon: break
    
    if np.sum(np.isnan(D)) > 0: raise Exception('NaN encountered in occupancy measure')
    return D


def irl_log_likelihood_and_grad(r, mdp, gamma, trajectories, horizon=None):
    
    V = compute_value_boltzmann(mdp, gamma, r, horizon=horizon)
    # IRL log likelihood term
    L_D = compute_irl_log_likelihood(mdp, gamma, trajectories, V, r)
    
    # IRL log likelihood gradient w.r.t reward
    policy = compute_policy(mdp, gamma, reward=r, V=V) 
    mu_hat, init_s = compute_s_a_visitations(mdp, gamma, trajectories)
    
    D = compute_D(mdp, gamma, V, policy, init_s, horizon=horizon)
    
    dL_D_dr = np.sum(mu_hat,1) - D
    
    return L_D, -dL_D_dr


def main():
 
    env = FrozenLakeEnvMultigoal(goal=2)
    env.seed(0);  prng.seed(10)

    NUM_ITER = 75
    gamma = 0.99

    horizon = 15
    mdp1 = MDP(FrozenLakeEnvMultigoal(is_slippery=False, goal=1))

    r1 = np.zeros(64)
    r1[63] = 1.0

    policy1 = compute_policy(mdp1, gamma, r1, threshold=1e-8, horizon=None)
    trajectories1 = generate_trajectories(mdp1, policy1, T=15, num_traj=200)

    epochs = 30
    r = np.random.rand(64)
    #r[63] = 1
    print(r)

    for i in range(epochs):
        cost, grad = irl_log_likelihood_and_grad(r, mdp1, gamma,  trajectories1, horizon=None)
        r = r - 0.1*grad
        print(cost)


if __name__ == "__main__":
    main()