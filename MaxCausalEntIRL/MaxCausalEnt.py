from frozen_lake import *
import numpy as np, gym
from gym.spaces import prng

def max_causal_ent_irl(mdp, gamma, trajectories, epochs=1, learning_rate=0.2, 
                       r = None, horizon=None):
    """
    Finds the reward vector that maximizes the log likelihood of the expert 
    trajectories via gradient descent.
    
    The gradient is the difference between the mean empirical state visitation 
    counts computed from the expert trajectories and the occupancy measure of 
    the MDP under a policy induced by the reward vector.

    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    gamma : float 
        Discount factor; 0<=gamma<=1.
    trajectories : 3D numpy array
        Expert trajectories. 
        Dimensions: [number of traj, timesteps in the traj, state and action].
    epochs : int
        Number of iterations gradient descent will run.
    learning_rate : float
        Learning rate for gradient descent.
    r : 1D numpy array
        Initial reward vector with the length equal to the #states in the MDP.
    horizon : int
        Horizon for the finite horizon version of value iteration.
    Returns
    -------
    1D numpy array
        Reward vector computed with Maximum Causal Entropy algorithm from 
        the expert trajectories.

    Note
    -------
    Following the Levine implementation, the state features are assumed to 
    be one-hot encodings of the state. If this is not the case, reward 
    would have to have the shape (feature.shape[0]), and the gradient of the 
    IRL log likelihood would be a dot product of the current expression 
    for dL_dr with the feature matrix.
    """    

    # Compute the empirical state-action visitation counts and the probability 
    # of a trajectory starting in state s from the expert trajectories.
    sa_visit_count, P_0 = compute_s_a_visitations(mdp, gamma, trajectories)
    
    if r is None:
        r = np.random.rand(mdp.nS)

    for i in range(epochs):
        V = compute_value_boltzmann(mdp, gamma, r, horizon=horizon)
        
        # Compute the Boltzmann policy \pi_{s,a} = \exp(Q_{s,a} - V_s) 
        policy = compute_policy(mdp, gamma, r=r, V=V) 
        
        # IRL log likelihood term: 
        # L = 0; for all traj: for all (s, a) in traj: L += Q[s,a] - V[s]
        L = np.sum(sa_visit_count * np.log(policy))
        
        # The expected number of times policy π visits state s in a given number of timesteps.
        D = compute_D(mdp, gamma, policy, P_0, t_max=trajectories.shape[1])        

        # Mean state visitation count of expert trajectories
        # mean_s_visit_count[s] = ( \sum_{i,t} 1_{traj_s_{i,t} = s} ) / num_traj
        mean_s_visit_count = np.sum(sa_visit_count,1) / trajectories.shape[0]

        # IRL log likelihood gradient w.r.t reward. Corresponds to line 9 of 
        # Algorithm 2 from the MaxCausalEnt IRL paper 
        # http://www.cs.cmu.edu/~bziebart/publications/maximum-causal-entropy.pdf. 
        # Refer to the Note in this function.Minus sign to get the gradient 
        # of negative log likelihood, which we then minimize with GD.
        dL_dr = -(mean_s_visit_count - D)

        # Gradient descent
        r = r - learning_rate * dL_dr

        print('Epoch: ',i, 'log likelihood of all traj: ', L, 
            'average per traj step: ', 
            L/(trajectories.shape[0] * trajectories.shape[1]))
    return r


class MDP(object):
    """
    MDP object

    Attributes
    ----------
    self.nS : int
        Number of states in the MDP.
    self.nA : int
        Number of actions in the MDP.
    self.P : two-level dict of lists of tuples
        First key is the state and the second key is the action. 
        self.P[state][action] is a list of tuples (probability, nextstate, reward).
    self.T : 3D numpy array
        The transition probability matrix of the MDP. p(s'|s,a) = self.T[s,a,s']
    """
    def __init__(self, env):
        P, nS, nA, desc = MDP.env2mdp(env)
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means 
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
    described in Algorithm 9.2 of Ziebart's PhD thesis 
    http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf.

    Note that softmax(softmax(x1,x2), x3) = log(exp(x1) + exp(x2) + exp(x3))
    """
    max_x = np.amax((x1,x2))
    min_x = np.amin((x1,x2))
    return max_x + np.log(1+np.exp(min_x - max_x))

def compute_value_boltzmann(mdp, gamma, r, horizon = None, threshold=1e-4):
    """
    Find the optimal value function via value iteration with the max-ent 
    Bellman backup given at Algorithm 9.1 in Ziebart's PhD thesis 
    http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf.

    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    gamma : float 
        Discount factor; 0<=gamma<=1.
    r : 1D numpy array
        Initial reward vector with the length equal to the 
        number of states in the MDP.
    horizon : int
        Horizon for the finite horizon version of value iteration.
    threshold : float
        Convergence threshold.

    Returns
    -------
    1D numpy array
        Array of shape (mdp.nS), each V[s] is the value of state s under 
        the reward r and Boltzmann policy.
    """
    
    V = np.zeros(mdp.nS)
    
    # Running into numerical problems with r= [[0]*63, 1] for some reason. 
    # Critch, check this out?
    V = r * .99999

    t = 0
    diff = float("inf")
    while diff > threshold:
        V_prev = np.copy(V)
        for s in range(mdp.nS):
            # V_s_new = \log[\sum_a \exp(r_s + \gamma \sum_{s'} p(s'|s,a)V_{s'} )]
            for a in range(mdp.nA):
                # If-else statement is used to compute softmax correctly. 
                # If V[s] is initialized as 0 and only the expression from 
                # 'else' is used, there would be an additional e^0 in the sum.
                if a == 0:
                    # V[s] = r_s + \gamma \sum_{s'} p(s'|s,a)V_{s'}
                    V[s] = r[s] + gamma * np.dot(mdp.T[s, a, :], V_prev)  
                else:
                    # V[s] = log(exp(V[s]) + exp(r_s + \gamma \sum_{s'} p(s'|s,a)V_{s'}))
                    V[s] = softmax(V[s], 
                                   r[s] + gamma * np.dot(mdp.T[s, a, :], V_prev))
            
                if np.sum(np.isnan(V[s])) > 0: 
                    raise Exception('NaN encountered in value, iteration ', 
                                    t, 'state',s, ' action ', a)
                        
        diff = np.amax(abs(V_prev - V))
        
        t+=1
        if horizon is not None:
            if t==horizon: break
    return V


def compute_policy(mdp, gamma, r=None, V=None, horizon=None, threshold=1e-4):
    """
    Computes the Boltzmann policy \pi_{s,a} = \exp(Q_{s,a} - V_s).
    
    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    gamma : float 
        Discount factor; 0<=gamma<=1.
    r : 1D numpy array
        Initial reward vector with the length equal to the #states in the MDP.
    V : 1D numpy array
        Value of each of the states of the MDP.
    horizon : int
        Horizon for the finite horizon version of value iteration.
    threshold : float
        Convergence threshold.

    Returns
    -------
    2D numpy array
        Array of shape (mdp.nS, mdp.nA), each value p[s,a] is the probability of 
        taking action a in state s.
    """

    if r is None and V is None: 
        raise Exception('Cannot compute V: no reward provided')
    if V is None: V = compute_value_boltzmann(mdp, gamma, r, horizon, threshold)

    policy = np.zeros((mdp.nS, mdp.nA))
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            # This is exp(Q_{s,a} - V_s)
            policy[s,a] = np.exp(r[s] + np.dot(mdp.T[s, a,:], gamma * V) - V[s] )
    
    # Hack for finite horizon length to make the probabilities sum to 1:
    policy = policy / np.sum(policy, axis=1).reshape((mdp.nS, 1))

    if np.sum(np.isnan(policy)) > 0: raise Exception('NaN encountered in policy')
    
    return policy


def generate_trajectories(mdp, policy, timesteps=20, num_traj=50):
    """
    Generates trajectories in the MDP given a policy.
    """
    s = mdp.env.reset()
    
    trajectories = np.zeros([num_traj, timesteps, 2]).astype(int)
    
    for d in range(num_traj):
        for t in range(timesteps):
            action = np.random.choice(range(mdp.nA), p=policy[s, :])
            trajectories[d, t, :] = [s, action]
            s, _, _, _ = mdp.env.step(action)
        s = mdp.env.reset()
    
    return trajectories


def compute_s_a_visitations(mdp, gamma, trajectories):
    """
    Computes the empirical state-action visitation counts and the probability 
    of a trajectory starting in state s from the expert trajectories.
    
    Empirical state-action visitation counts:
    sa_visit_count[s,a] = \sum_{i,t} 1_{traj_s_{i,t} = s \wedge traj_a_{i,t} = a}

    P_0(s) -- probability that the trajectory will start in state s. 
    P_0[s] = \sum_{i,t} 1_{t = 0 \wedge traj_s_{i,t} = s}  / i
    Used in computing the occupancy measure of a MDP under a given policy.

    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    gamma : float 
        Discount factor; 0<=gamma<=1.
    trajectories : 3D numpy array
        Expert trajectories. 
        Dimensions: [number of traj, timesteps in the traj, state and action].

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


def compute_D(mdp, gamma, policy, P_0=None, t_max=None, threshold = 1e-6):
    """
    Computes occupancy measure of a MDP under a given time-constrained policy 
    -- the expected number of times that policy π visits state s in a given 
    number of timesteps.
    
    Described in Algorithm 9.3 of Ziebart's PhD thesis 
    http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf.

    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    gamma : float 
        Discount factor; 0<=gamma<=1.
    policy : 2D numpy array
        policy[s,a] is the probability of taking action a in state s.
    P_0 : 1D numpy array of shape (mdp.nS)
        i-th element is the probability that the traj will start in state i.
    t_max : int
        number of timesteps the policy is executed.

    Returns
    -------
    1D numpy array of shape (mdp.nS)
    """
    assert policy.shape == (mdp.nS, mdp.nA)   
    assert np.sum(np.isnan(policy)) == 0
        
    if P_0 is None: P_0 = np.ones(mdp.nS) / mdp.nS
    D_prev = np.copy(P_0)     
    
    t = 1
    diff = float("inf")
    while diff > threshold:
        
        # Line 6 of Algorithm 9.3: 
        # for all s: D[s] <- P_0[s]
        D = np.copy(P_0)

        for s in range(mdp.nS):
            for a in range(mdp.nA):
                # Line 9 of Algorithm 9.3:
                # for all s_prime reachable from s by taking a do:
                for p_sprime, s_prime, _ in mdp.P[s][a]:
                    # Line 10 of Algorithm 9.3:
                    D[s_prime] += D_prev[s] * policy[s, a] * p_sprime

        diff = np.amax(abs(D_prev - D))    
        D_prev = np.copy(D)
        
        if t_max is not None:
            t+=1
            if t==t_max: break
    
    if np.sum(np.isnan(D_prev)) > 0: 
        raise Exception('NaN encountered in occupancy measure')
    return D


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
    trajectories1 = generate_trajectories(mdp1, policy1, traj_len, num_traj=200)
    print('Generated ', trajectories1.shape[0],' traj of length ', traj_len)

    sa_visit_count, _ = compute_s_a_visitations(mdp1, gamma, trajectories1)
    print('Log likelihood of all traj under the policy generated from the original reward: ', 
        np.sum(sa_visit_count * np.log(policy1)), 
        'average per traj step: ', 
        np.sum(sa_visit_count * np.log(policy1)) / 
                    (trajectories1.shape[0] * trajectories1.shape[1]), '\n' )

    r = np.random.rand(mdp1.nS)
    print('Randomly initialized reward: ',r)

    r = max_causal_ent_irl(mdp1, gamma, trajectories1, epochs, learning_rate,
                           r = r, horizon=horizon)

    print('Final reward: ', r)

if __name__ == "__main__":
    main()