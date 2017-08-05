from frozen_lake import *
import numpy as np 
from gym.spaces import prng

def max_causal_ent_irl(mdp, gamma, trajectories, epochs=1, learning_rate=0.2, 
                       r = None, horizon=None, temperature=1):
    """
    Finds a reward vector that maximizes the log likelihood of the given expert 
    trajectories, modelling the expert as a Boltzmann rational agent with the 
    given temperature. By [citation], this is equivalent to finding a reward 
    vector giving rise to a Boltzmann rational policy whose expected state 
    visitation count matches the visitation counts of the given expert 
    trajectories.
    
    The gradient is the difference between the mean state visitation counts 
    computed from the expert trajectories and the occupancy measure of the MDP 
    under a policy induced by the reward vector.

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
    """    

    # Compute the state-action visitation counts and the probability 
    # of a trajectory starting in state s from the expert trajectories.
    sa_visit_count, P_0 = compute_s_a_visitations(mdp, gamma, trajectories)
    
    if r is None:
        r = np.random.rand(mdp.nS)

    for i in range(epochs):
        V, Q = compute_value_boltzmann(mdp, gamma, r, horizon = horizon, 
                                       temperature=temperature)
        
        # Compute the Boltzmann rational policy \pi_{s,a} = \exp(Q_{s,a} - V_s) 
        policy = compute_policy(mdp, gamma, r, V, Q, horizon=horizon, 
                                temperature=temperature) 
        
        # IRL log likelihood term: 
        # L = 0; for all traj: for all (s, a) in traj: L += Q[s,a] - V[s]
        L = np.sum(sa_visit_count * (Q - V.reshape((mdp.nS,1))))
        
        # The expected #times policy π visits state s in a given #timesteps.
        D = compute_D(mdp, gamma, policy, P_0, t_max=trajectories.shape[1])        

        # Mean state visitation count of expert trajectories
        # mean_s_visit_count[s] = ( \sum_{i,t} 1_{traj_s_{i,t} = s}) / num_traj
        mean_s_visit_count = np.sum(sa_visit_count,1) / trajectories.shape[0]

        # IRL log likelihood gradient w.r.t reward. Corresponds to line 9 of 
        # Algorithm 2 from the MaxCausalEnt IRL paper 
        # www.cs.cmu.edu/~bziebart/publications/maximum-causal-entropy.pdf. 
        # Refer to the Note in this function. Minus sign to get the gradient 
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
        self.P[state][action] is a list of tuples (prob, nextstate, reward).
    self.T : 3D numpy array
        The transition prob matrix of the MDP. p(s'|s,a) = self.T[s,a,s']
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


def softmax(x):
    '''
    Computes log(\sum_i exp(x_i))
    '''
    if x.shape[0] == 1: return x
    
    def softmax_2_arg(x1,x2):
        """ 
        Numerically stable computation of log(exp(x1) + exp(x2))
        described in Algorithm 9.2 of Ziebart's PhD thesis 
        http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf.
        """
        max_x = np.amax((x1,x2))
        min_x = np.amin((x1,x2))
        return max_x + np.log(1+np.exp((min_x - max_x)))
    
    sm = softmax_2_arg(x[0],x[1])
    # Use the following property of softmax_2_arg:
    # softmax_2_arg(softmax_2_arg(x1,x2),x3) = log(exp(x1) + exp(x2) + exp(x3))
    # which is true since
    # log(exp(log(exp(x1) + exp(x2))) + exp(x3)) = log(exp(x1) + exp(x2) + exp(x3))
    for (i,x_i) in enumerate(x):
        if i>1: sm = softmax_2_arg(sm, x_i)
    return sm


def compute_value_boltzmann(mdp, gamma, r, horizon = None, threshold=1e-4, 
                            temperature=1):
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
    2D numpy array
        Array of shape (mdp.nS, mdp.nA), each Q[s,a] is the value of 
        state-action pair [s,a] under the reward r and Boltzmann policy.
    """
    
    V = np.copy(r)
    Q = np.tile(r, (mdp.nA,1)).T

    t = 0
    diff = float("inf")
    while diff > threshold:
        V_prev = np.copy(V)
        for s in range(mdp.nS):
            for a in range(mdp.nA):
                # Q[s,a] = (r_s + \gamma \sum_{s'} p(s'|s,a)V_{s'}) / temperature
                Q[s,a] = (r[s] + gamma * np.dot(mdp.T[s, a, :], V_prev)) / temperature
            
            # V_s = \log(\sum_a exp(Q_sa))    
            V[s] = softmax(Q[s,:])
            
            if np.sum(np.isnan(V[s])) > 0: 
                raise Exception('NaN encountered in VI, t=',t, 's=',s)
        
        # Normalize
        Q = Q - V.mean()
        V = V - V.mean()
        print(V)

        diff = np.amax(abs(V_prev - V))
        
        t+=1
        if horizon is not None:
            if t==horizon: break
    return (V, Q)


def compute_policy(mdp, gamma, r=None, V=None, Q=None, horizon=None, 
                   threshold=1e-4, temperature = 1):
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
        Array of shape (mdp.nS, mdp.nA), each value p[s,a] is the probability 
        of taking action a in state s.
    """

    if V is None or Q is None: 
        V, Q = compute_value_boltzmann(mdp, gamma, r, horizon, 
                                    threshold, temperature)

    policy = np.zeros((mdp.nS, mdp.nA))
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            # This is exp(Q_{s,a} - V_s)
            policy[s,a] = np.exp(Q[s,a] - V[s])
    
    # Hack for finite horizon length to make the probabilities sum to 1:
    policy = policy / np.sum(policy, axis=1).reshape((mdp.nS, 1))

    if np.sum(np.isnan(policy)) > 0: 
        raise Exception('NaN encountered in policy')
    
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
    Computes the state-action visitation counts and the probability 
    of a trajectory starting in state s from the expert trajectories.
    
    State-action visitation counts:
    sa_visit_count[s,a] = \sum_{i,t} 1_{traj_s_{i,t} = s AND traj_a_{i,t} = a}

    P_0(s) -- probability that the trajectory will start in state s. 
    P_0[s] = \sum_{i,t} 1_{t = 0 AND traj_s_{i,t} = s}  / i
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
    temp_exp = 1
    temp_irl = 1
    
    learning_rate = 0.1
    epochs = 30
    
    gamma = 1
    horizon = 200
    traj_len = 15

    env = FrozenLakeEnvMultigoal(goal=2)
    env.seed(0);  prng.seed(10)
    mdp1 = MDP(FrozenLakeEnvMultigoal(is_slippery=False, goal=1))    

    r1 = np.zeros(mdp1.nS)
    r1[-1] = 1
    print('Reward used to generate expert trajectories: ', r1)
    
    V1, Q1 = compute_value_boltzmann(mdp1, gamma, r1, horizon = horizon, 
                                       temperature=temp_exp)
    policy1 = compute_policy(mdp1, gamma, r1, V1, Q1, 
                             threshold=1e-8, horizon=horizon)
    trajectories1 = generate_trajectories(mdp1, policy1, traj_len, 200)
    print('Generated ', trajectories1.shape[0],' traj of length ', traj_len)

    sa_visit_count, _ = compute_s_a_visitations(mdp1, gamma, trajectories1)
    #print('Log likelihood of all traj under the policy generated', 
    #      'from the original reward: ', 
    #      np.sum(sa_visit_count * (Q1 - V1.reshape((mdp1.nS,1)))), 
    #      'average per traj step: ', 
    #      np.sum(sa_visit_count * (Q1 - V1.reshape((mdp1.nS,1))) / 
    #            (trajectories1.shape[0] * trajectories1.shape[1]), '\n' ))

    r = np.random.rand(mdp1.nS)
    print('Randomly initialized reward: ',r)

    r = max_causal_ent_irl(mdp1, gamma, trajectories1, epochs, learning_rate,
                           r = r, horizon=horizon, temperature=temp_irl)

    print('Final reward: ', r)

if __name__ == "__main__":
    main()