import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MaxCausalEnt import *

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
        P, nS, nA, desc = MDP.env2mdp(env)
        
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
        for (p,s,_) in mdp.P[s][a]: 
            if len(mdp.P[s][a])==1:
                return s
            else:
                raise Warning("Non-deterministic transactions not implemented")
            
    
    trajectories = np.zeros([num_traj, timesteps, 2]).astype(int)
    
    for d in range(num_traj):
        for t in range(timesteps):
            action = np.random.choice(range(mdp.nA), p=policy[s, :])
            trajectories[d, t, :] = [s, action]
            s = mdp_step(mdp, s, action)
        s = mdp.env.reset()
    
    return trajectories

def bar(t_expert = 1,
         gamma = 0.999,
         horizon = 200,
         n_traj = 10000,
         traj_len = 12,
         mdp_one_time_reward = False):

    env = FrozenLakeEnvMultigoal(goal=1)
    env.seed(0);  prng.seed(0); np.random.seed(0)
    if mdp_one_time_reward == True:
        mdp = MDP_one_time_r(FrozenLakeEnvMultigoal(is_slippery=False, goal=1))
    else:
        mdp = MDP(FrozenLakeEnvMultigoal(is_slippery=False, goal=1))

    # The true reward 
    r_expert = np.zeros(mdp.nS)
    r_expert[24] = 1
    
    # Compute the Boltzmann rational expert policy from the given true reward.
    V, Q = compute_value_boltzmann(mdp, gamma, r_expert, horizon, t_expert)
    policy_expert = compute_policy(mdp, gamma, r_expert, V, Q, horizon, 
                                   temperature=t_expert)
    
    # Generate expert trajectories using the given expert policy.
    trajectories = generate_trajectories(mdp, policy_expert, traj_len, n_traj)
    
    # Compute and print various stats of the generated expert trajectories.
    sa_visit_count, _ = compute_s_a_visitations(mdp, gamma, trajectories)
    return (np.sum(sa_visit_count, axis=1) / n_traj)[24]


temp = 1.5**(np.arange(46) - 41)
temp += -np.amin(temp)

horizons = [12, 40, 80, 200]
cols = ('one time r','t','h','average return per traj')
d = pd.DataFrame(columns=cols)
i=0
for mdp_one_time_r in [True, False]:
    for t in temp:
        for h in horizons:
            r = bar(t, horizon=h, mdp_one_time_reward=mdp_one_time_r)
            d.loc[i] = [mdp_one_time_r, t,h,r]
            i+=1


d.to_pickle('df_t_h_r.pkl')
d =  pd.read_pickle('df_t_h_r_one_time_low_t.pkl')

for i in horizons:
    d.loc[(d['h'] == i) & (d['one time r'] == True)].plot(x='t', 
          y='average return per traj', title='h='+str(i), 
          ylim = (0,1.1), xlim = (0,0.0005))

for i in horizons:
    d.loc[(d['h'] == i) & (d['one time r'] == False)].plot(x='t', 
          y='average return per traj', title='h='+str(i), 
          ylim = (0,8), xlim = (0,5))



print(d.loc[(d['h'] == 20) & (d['one time r'] == False)])


