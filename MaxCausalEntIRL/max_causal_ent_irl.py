import numpy as np 
from frozen_lake import FrozenLakeEnvMultigoal
from mdps import MDP, MDPOneTimeR, generate_trajectories
from value_iter_and_policy import vi_boltzmann, compute_policy_boltzmann 
from value_iter_and_policy import vi_rational, compute_policy_rational

def max_causal_ent_irl(mdp, trajectories, gamma=1, horizon=None, temperature=1, 
                       epochs=1, learning_rate=0.2, r=None):
    '''
    Finds a reward vector that maximizes the log likelihood of the given expert 
    trajectories, modelling the expert as a Boltzmann rational agent with the 
    given temperature. By [citation], this is equivalent to finding a reward 
    vector giving rise to a Boltzmann rational policy whose expected state 
    visitation count matches the visitation counts of the given expert 
    trajectories.
    
    The gradient of the log likelihood of the expert trajectories w.r.t the 
    reward is the difference between the mean state visitation counts 
    computed from the expert trajectories and the occupancy measure of the MDP 
    under a policy induced by the reward vector.

    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    gamma : float 
        Discount factor; 0<=gamma<=1.
    horizon : int
        Horizon for the finite horizon version of value iteration.
    trajectories : 3D numpy array
        Expert trajectories. 
        Dimensions: [number of traj, timesteps in the traj, state and action].
    epochs : int
        Number of iterations gradient descent will run.
    learning_rate : float
        Learning rate for gradient descent.
    r : 1D numpy array
        Initial reward vector with the length equal to the #states in the MDP.
    Returns
    -------
    1D numpy array
        Reward vector computed with Maximum Causal Entropy algorithm from 
        the expert trajectories.
    '''    
    
    # Compute the state-action visitation counts and the probability 
    # of a trajectory starting in state s from the expert trajectories.
    sa_visit_count, P_0 = compute_s_a_visitations(mdp, gamma, trajectories)
    
    if r is None:
        r = np.random.rand(mdp.nS)

    for i in range(epochs):
        V, Q = vi_boltzmann(mdp, gamma, r, horizon, temperature)
        
        # Compute the Boltzmann rational policy \pi_{s,a} = \exp(Q_{s,a} - V_s) 
        policy = compute_policy_boltzmann(mdp, V, Q, temperature)
        
        # IRL log likelihood term: 
        # L = 0; for all traj: for all (s, a) in traj: L += Q[s,a] - V[s]
        L = np.sum(sa_visit_count * (Q - V))
        
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

        if i%3==0: 
            print('Epoch: {} log likelihood of all traj: {}'.format(i,L), 
                  ', average per traj step: {}'.format(
                  L/(trajectories.shape[0] * trajectories.shape[1])))
        
    return r


def compute_s_a_visitations(mdp, gamma, trajectories):
    '''
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
    '''

    s_0_count = np.zeros(mdp.nS)
    sa_visit_count = np.zeros((mdp.nS, mdp.nA))
    
    for traj in trajectories:
        # traj[0][0] is the state of the first timestep of the trajectory.
        s_0_count[traj[0][0]] += 1
        for (s, a) in traj:
            sa_visit_count[s, a] += 1
      
    # Count into probability        
    P_0 = s_0_count / trajectories.shape[0]
    
    return sa_visit_count, P_0


def compute_D(mdp, gamma, policy, P_0=None, t_max=None, threshold=1e-6):
    '''
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
    '''

    if P_0 is None: P_0 = np.ones(mdp.nS) / mdp.nS
    D_prev = np.copy(P_0)     
    
    t = 1
    diff = float("inf")
    while diff > threshold:
        
        # for all s: D[s] <- P_0[s]
        D = np.copy(P_0)

        for s in range(mdp.nS):
            for a in range(mdp.nA):
                # for all s_prime reachable from s by taking a do:
                for p_sprime, s_prime, _ in mdp.P[s][a]:
                    D[s_prime] += D_prev[s] * policy[s, a] * p_sprime

        diff = np.amax(abs(D_prev - D))    
        D_prev = np.copy(D)
        
        if t_max is not None:
            t+=1
            if t==t_max: break
    
    if np.sum(np.isnan(D_prev)) > 0: 
        raise Exception('NaN encountered in occupancy measure')
    return D


def main(t_expert=1e-6,
         t_irl=1e-6,
         gamma=1,
         horizon=10,
         n_traj=200,
         traj_len=10,
         learning_rate=0.05,
         epochs=31):
    '''
    Demonstrates the usage of the implemented MaxCausalEnt IRL algorithm. 
    
    First a number of expert trajectories is generated using a true reward 
    giving rise to the Boltzmann rational expert policy with temperature t_exp. 
    
    Hereafter the max_causal_ent_irl() function is used to find a reward vector
    that maximizes the log likelihood of the generated expert trajectories, 
    modelling the expert as a Boltzmann rational agent with temperature t_irl.
    
    Parameters
    ----------
    t_expert : float
        Temperature of the Boltzmann rational expert policy.
    t_irl : float
        Temperature of the Boltzmann rational policy the IRL algorithm assumes
        the expert followed when generating the trajectories.
    gamma : float 
        Discount factor; 0<=gamma<=1.
    horizon : int
        Horizon for the finite horizon version of value iteration subroutine of
        MaxCausalEnt IRL algorithm.
    n_traj : int
        Number of expert trajectories generated.
    traj_len : int
        Number of timesteps in each of the expert trajectories.
    learning_rate : float
        Learning rate for gradient descent in the MaxCausalEnt IRL algorithm.
    epochs : int
        Number of gradient descent steps in the MaxCausalEnt IRL algorithm.
    '''
    np.random.seed(0)
    mdp = MDPOneTimeR(FrozenLakeEnvMultigoal(is_slippery=False, goal=1))    

    # The true reward 
    r_expert = np.zeros(mdp.nS)
    r_expert[24] = 1
    
    # Compute the Boltzmann rational expert policy from the given true reward.
    if t_expert>0:
        V, Q = vi_boltzmann(mdp, gamma, r_expert, horizon, t_expert)
        policy_expert = compute_policy_boltzmann(mdp, V, Q, t_expert)
    if t_expert==0:
        V, Q = vi_rational(mdp, gamma, r_expert, horizon)
        policy_expert = compute_policy_rational(Q)
        
    # Generate expert trajectories using the given expert policy.
    trajectories = generate_trajectories(mdp, policy_expert, traj_len, n_traj)
    
    # Compute and print various stats of the generated expert trajectories.
    sa_visit_count, _ = compute_s_a_visitations(mdp, gamma, trajectories)
    log_likelihood = np.sum(sa_visit_count * (Q - V))
    print('Generated {} traj of length {}'.format(n_traj, traj_len))
    print('Log likelihood of all traj under the policy generated ', 
          'from the true reward: {}, \n average per traj step: {}'.format(
           log_likelihood, log_likelihood / (n_traj * traj_len)))
    print('Average return per expert trajectory: {} \n'.format(
            np.sum(np.sum(sa_visit_count, axis=1)*r_expert) / n_traj))

    # Find a reward vector that maximizes the log likelihood of the generated 
    # expert trajectories.
    r = max_causal_ent_irl(mdp, trajectories, gamma, horizon, t_irl, epochs, 
                           learning_rate)
    print('Final reward: ', r)

if __name__ == "__main__":
    main()