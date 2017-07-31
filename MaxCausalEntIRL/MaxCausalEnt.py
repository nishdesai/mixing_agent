from frozen_lake import *
import numpy as np, numpy.random as nr, gym
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
        return ({s : {a : [tup[:3] for tup in tups] 
                    for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, 
                env.nS, env.nA, env.desc)
    
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
    """ 
    Numerically stable computation of log(exp(x1) + exp(x2))
    described in Algorithm 9.2 of Ziebart's PhD thesis.

    Note that softmax(softmax(x1,x2), x3) = log(exp(x1) + exp(x2) + exp(x3))
    """
    max_x = np.amax((x1,x2))
    min_x = np.amin((x1,x2))
    return max_x + np.log(1+np.exp(min_x - max_x))

def compute_value_boltzmann(mdp, gamma, r, horizon = None, threshold=1e-4):
    """
    Find the optimal value function via value iteration with the max-ent Bellman backup 
    given at Algorithm 9.1 in Ziebart's PhD thesis and in 
    https://graphics.stanford.edu/projects/gpirl/gpirl_supplement.pdf.

    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    gamma : float 
        Discount factor; 0<=gamma<=1.
    r : 1D numpy array
        Initial reward vector with the length equal to the number of states in the MDP.
    horizon : int
        Horizon for the finite horizon versions of value iteration and occupancy measure computations.
    threshold : float
        Convergence threshold.

    Returns
    -------
    1D numpy array
        Array of shape (mdp.nS), each V[s] is the value of state s under the reward r 
        and Boltzmann policy.
    """
    
    V = np.zeros(mdp.nS)
    
    # Running into numerical problems with r= [[0]*63, 1] for some reason. Critch, check this out?
    V = r * .99999

    t = 0
    diff = float("inf")
    while diff > threshold:
        V_prev = np.copy(V)
        for s in range(mdp.nS):
            # V_s_new = \log[  \sum_a \exp(  r_s + \gamma \sum_{s'} p(s'|s,a)V_{s'}   )  ]
            for a in range(mdp.nA):
                # If-else statement is used to compute softmax correctly. If V[s] is initialized as 0 and
                # only the expression from 'else' is used, there would be an additional e^0 in the sum.
                if a == 0:
                    # V[s] = r_s + \gamma \sum_{s'} p(s'|s,a)V_{s'}
                    V[s] = r[s] + gamma * np.dot(mdp.T[s, a, :], V_prev)  
                else:
                    # V[s] = log(   exp(V[s]) + exp(r_s + \gamma \sum_{s'} p(s'|s,a)V_{s'})   )
                    V[s] = softmax(V[s], r[s] + gamma * np.dot(mdp.T[s, a, :], V_prev))
            
                if np.sum(np.isnan(V[s])) > 0: 
                    raise Exception('NaN encountered in value, iteration ', t, 'state',s, ' action ', a)
                        
        diff = np.amax(abs(V_prev - V))
        
        t+=1
        if horizon is not None:
            if t==horizon: break
    
    return V


def compute_policy(mdp, gamma, r=None, V=None, horizon=None, threshold=1e-4):
    """
    Computes the Boltzmann policy \pi_{s,a} = \exp(Q_{s,a} - V_s) from the value function.
    
    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    gamma : float 
        Discount factor; 0<=gamma<=1.
    r : 1D numpy array
        Initial reward vector with the length equal to the number of states in the MDP.
    V : 1D numpy array
        Value of each of the states of the MDP.
    horizon : int
        Horizon for the finite horizon versions of value iteration and occupancy measure computations.
    threshold : float
        Convergence threshold.

    Returns
    -------
    2D numpy array
        Array of shape (mdp.nS, mdp.nA), each value p[s,a] is the probability of 
        taking action a in state s.
    """

    if r is None and V is None: raise Exception('Cannot compute V: no reward provided')
    if V is None: V = compute_value_boltzmann(mdp, gamma, r, horizon=horizon, threshold=threshold)

    policy = np.zeros((mdp.nS, mdp.nA))
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            # This is exp(Q_{s,a} - V_s)
            policy[s,a] = np.exp(r[s] + np.dot(mdp.T[s, a,:], gamma * V) - V[s] )
    
    # Hack for finite horizon length to make the probabilities sum to 1:
    # print(np.sum(policy, axis=1))
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


def compute_s_a_visitations(mdp, gamma, trajectories):
    """
    Computes the empirical state and state-action visitation frequencies from 
    the expert trajectories
    
    Empirical state-action visitation frequencies:
    sa_visit_count[s,a] = \sum_{i,t} 1_{traj_s_{i,t} = s \wedge traj_a_{i,t} = a}

    P_0(s) for initialization of the algorithm computing the occupancy measure 
    of a MDP under a given policy:
    P_0[s] = \sum_{i,t} 1_{t = 0 \wedge traj_s_{i,t} = s}  / i

    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    gamma : float 
        Discount factor; 0<=gamma<=1.
    trajectories : 3D numpy array
        Expert trajectories. 
        Dimensions: [number of trajectories, timesteps in the trajectory, state and action].

    Returns
    -------
    (2D numpy array, 1D numpy array)
        Arrays of shape (mdp.nS, mdp.nA) and (mdp.nS).
    """

    s_0_count = np.zeros(mdp.nS)
    sa_visit_count = np.zeros((mdp.nS, mdp.nA))
    
    for traj in trajectories:
        # traj[0][0] is the state of the first timestep of the trajectory.
        s_0_count[traj[0][0]] += 1
        for (s, a) in traj:
            sa_visit_count[s, a] += 1
      
    # Count into probability        
    P_0 = s_0_count / trajectories.shape[0]
    
    return(sa_visit_count, P_0)


def compute_D(mdp, gamma, V, policy, init_s=None, horizon=None, D_prev = None, threshold = 1e-6):
    """
    Computes occupancy measure of a MDP under a given policy -- 
    the expected discounted number of times that policy π visits state s.
    
    Described in Algorithm 9.3 of Ziebart's PhD thesis.
    """
    assert V.shape[0] == mdp.nS
    assert policy.shape == (mdp.nS, mdp.nA)   
    
    assert np.sum(np.isnan(V)) == 0
    assert np.sum(np.isnan(policy)) == 0
        
    if init_s is None: init_s = np.ones(mdp.nS) / mdp.nS
    if D_prev is None: D_prev = np.copy(init_s)     
    
    t = 1
    

    diff = float("inf")
    while diff > threshold:
        
        D = np.copy(init_s)

        for s in range(mdp.nS):
            for a in range(mdp.nA):
                for p_sprime, s_prime, _ in mdp.P[s][a]:

                    D[s_prime] += (policy[s, a] * ( gamma * p_sprime * D_prev[s]))

        diff = np.amax(abs(D_prev - D))    
        D_prev = np.copy(D)
        
        if horizon is not None:
            t+=1
            if t==horizon: break
    
    if np.sum(np.isnan(D_prev)) > 0: raise Exception('NaN encountered in occupancy measure')
    return D


def max_causal_ent_irl(mdp, gamma, trajectories, epochs=1, learning_rate=0.2, r = None, horizon=None):
    """
    Finds the reward vector that maximizes the log likelihood of the expert trajectories via gradient descent.
    
    The gradient is the difference between the empirical state visitation frequencies computed from the 
    expert trajectories and the occupancy measure of the MDP under a policy induced by the reward vector.

    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    gamma : float 
        Discount factor; 0<=gamma<=1.
    trajectories : 3D numpy array
        Expert trajectories. 
        Dimensions: [number of trajectories, timesteps in the trajectory, state and action].
    epochs : int
        Number of iterations gradient descent will run.
    learning_rate : float
        Learning rate for gradient descent.
    r : 1D numpy array
        Initial reward vector with the length equal to the number of states in the MDP.
    horizon : int
        Horizon for the finite horizon versions of value iteration and occupancy measure computations.

    Returns
    -------
    1D numpy array
        Reward vector computed with Maximum Causal Entropy algorithm from the expert trajectories.
    """

    

    sa_visit_count, init_s = compute_s_a_visitations(mdp, gamma, trajectories)
    
    if r is not None:
        r = np.random.rand(mdp.nS)

    for i in range(epochs):
        V = compute_value_boltzmann(mdp, gamma, r, horizon=horizon)
        policy = compute_policy(mdp, gamma, r=r, V=V) 
        
        # IRL log likelihood term
        L = np.sum(sa_visit_count * np.log(policy))
        
        # IRL log likelihood gradient w.r.t reward, inverted for descent
        D = compute_D(mdp, gamma, V, policy, init_s, horizon=trajectories.shape[1])        

        # Minus sign to get the gradient of negative log likelihood, which we then minimize wiht GD
        dL_dr = -(np.sum(sa_visit_count,1) / trajectories.shape[0] - D)

        # Gradient descent
        r = r - learning_rate * dL_dr

        print('Epoch: ',i, 'log likelihood of all traj: ', L, 'average per traj step: ', L/(trajectories.shape[0] * trajectories.shape[1]))

    return r


def main():

    learning_rate = 0.1
    epochs = 20
    
    gamma = 1
    horizon = 200
    traj_len = 15

    env = FrozenLakeEnvMultigoal(goal=2)
    env.seed(0);  prng.seed(10)
    mdp1 = MDP(FrozenLakeEnvMultigoal(is_slippery=False, goal=1))
    r1 = np.zeros(mdp1.nS)
    r1[-1] = 1
    print('Reward used to generate expert trajectories: ', r1)

    policy1 = compute_policy(mdp1, gamma, r1, threshold=1e-8, horizon=horizon)
    trajectories1 = generate_trajectories(mdp1, policy1, T=traj_len, num_traj=200)
    print('Generated ', trajectories1.shape[0],' trajectories of length ', traj_len)

    sa_visit_count, _ = compute_s_a_visitations(mdp1, gamma, trajectories1)
    print('Log likelihood of the trajectories under the policy generated from the original reward: ', np.sum(sa_visit_count * np.log(policy1)), 
        'average per traj step: ', np.sum(sa_visit_count * np.log(policy1)) / (trajectories1.shape[0] * trajectories1.shape[1]), '\n' )

    r = np.random.rand(mdp1.nS)
    print('Randomly initialized reward: ',r)

    r = max_causal_ent_irl(mdp=mdp1, gamma=gamma, trajectories=trajectories1, 
                        epochs=epochs, learning_rate=learning_rate, r = r, horizon=horizon)

    print('Final reward: ', r)

if __name__ == "__main__":
    main()