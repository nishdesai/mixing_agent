import numpy as np
from frozen_lake import FrozenLakeEnvMultigoal
from gym.spaces import prng


class MDP(object):
    '''
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
    '''
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
        '''Return a matrix with index S,A,S' -> P(S'|S,A)'''
        T = np.zeros([self.nS, self.nA, self.nS])
        for s in range(self.nS):
            for a in range(self.nA):
                transitions = self.P[s][a]
                s_a_s = {t[1]:t[0] for t in transitions}
                for s_prime in range(self.nS):
                    if s_prime in s_a_s:
                        T[s, a, s_prime] = s_a_s[s_prime]
        return T


class MDP_one_time_r(object):
    '''
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
    '''
    def __init__(self, env):
        P, nS, nA, desc=MDP.env2mdp(env)
        
        self.P = P
        self.P.update({nS-1:{0:[(1.0,nS,0.0)], 1:[(1.0,nS,0.0)], 2:[(1.0,nS,0.0)], 3:[(1.0,nS,0.0)]}})
        self.P.update({nS:{0:[(1.0,nS,0.0)], 1:[(1.0,nS,0.0)], 2:[(1.0,nS,0.0)], 3:[(1.0,nS,0.0)]}})
        self.nS = nS+1
        
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means 
        self.env = env
        self.T = self.get_transition_matrix()

    def env2mdp(env):
        return ({s : {a : [tup[:3] for tup in tups] 
                for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, 
                env.nS, env.nA, env.desc)
    
    def get_transition_matrix(self):
        '''Return a matrix with index S,A,S' -> P(S'|S,A)'''
        T = np.zeros([self.nS, self.nA, self.nS])
        for s in range(self.nS):
            for a in range(self.nA):
                transitions = self.P[s][a]
                s_a_s = {t[1]:t[0] for t in transitions}
                for s_prime in range(self.nS):
                    if s_prime in s_a_s:
                        T[s, a, s_prime] = s_a_s[s_prime]
        return T


def generate_trajectories(mdp, policy, timesteps=20, num_traj=50):
    '''
    Generates trajectories in the MDP given a policy.
    '''
    s = mdp.env.reset()
    
    def mdp_step(mdp, s, a):
        if len(mdp.P[s][a])==1:
            return mdp.P[s][a][0][1]
        else:
            p_s_sa = np.asarray(mdp.P[s][a])[:,0]
            next_state_index = np.random.choice(len(p_s_sa), 1, p_s_sa)
            return mdp.P[s][a][next_state_index][1]
    
    trajectories = np.zeros([num_traj, timesteps, 2]).astype(int)
    
    for d in range(num_traj):
        for t in range(timesteps):
            action = np.random.choice(range(mdp.nA), p=policy[s, :])
            trajectories[d, t, :] = [s, action]
            s = mdp_step(mdp, s, action)
        s = mdp.env.reset()
    
    return trajectories