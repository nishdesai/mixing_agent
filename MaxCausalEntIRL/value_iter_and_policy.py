import numpy as np


def vi_boltzmann(mdp, gamma, r, horizon=None,  temperature=1, 
                            threshold=1e-16, use_mellowmax=False):
    '''
    Finds the optimal state and state-action value functions via value 
    iteration with the "soft" max-ent Bellman backup:
    
    Q_{sa} = r_s + gamma * \sum_{s'} p(s'|s,a)V_{s'}
    V'_s = temperature * log(\sum_a exp(Q_{sa}/temperature))

    Computes the Boltzmann rational policy 
    \pi_{s,a} = exp((Q_{s,a} - V_s)/temperature).
    
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
        Array of shape (mdp.nS, 1), each V[s] is the value of state s under 
        the reward r and Boltzmann policy.
    2D numpy array
        Array of shape (mdp.nS, mdp.nA), each Q[s,a] is the value of 
        state-action pair [s,a] under the reward r and Boltzmann policy.
    2D numpy array
        Array of shape (mdp.nS, mdp.nA), each value p[s,a] is the probability 
        of taking action a in state s.
    '''
    #Value iteration    
    V = np.copy(r)
    t = 0
    diff = float("inf")
    while diff > threshold:
        V_prev = np.copy(V)
        
        # Q[s,a] = (r_s + gamma * \sum_{s'} p(s'|s,a)V_{s'})
        Q = r.reshape((-1,1)) + gamma * np.dot(mdp.T, V_prev)
        if use_mellowmax:
            # V_s = log(\sum_a exp(Q_sa) / nA)
            V = mellowmax(Q, temperature)
        else:
            # V_s = log(\sum_a exp(Q_sa))
            V = softmax(Q, temperature)

        diff = np.amax(abs(V_prev - V))
        
        t+=1
        if t<horizon and gamma==1 and not use_mellowmax:
            # When \gamma=1, the backup operator is equivariant under adding 
            # a constant to all entries of V, so we can translate min(V) 
            # to be 0 at each step of the softmax value iteration without 
            # changing the policy it converges to, and this fixes the problem 
            # where log(nA) keep getting added at each iteration.
            V = V - np.amin(V)
        if horizon is not None:
            if t==horizon: break
    
    V = V.reshape((-1, 1))
    
    # Compute policy
    expt = lambda x: np.exp(x/temperature)
    tlog = lambda x: temperature * np.log(x)

    if use_mellowmax:
        # exp((Q_{s,a} - V_s - t*log(nA))/t)
        policy = expt(Q - V - tlog(Q.shape[1]))
    else:
        # exp((Q_{s,a} - V_s)/t)
        policy = expt(Q - V)
        
    return V, Q, policy



def vi_rational(mdp, gamma, r, horizon=None, threshold=1e-16):
    '''
    Finds the optimal state and state-action value functions via value 
    iteration with the Bellman backup.
    
    Computes the rational policy \pi_{s,a} = \argmax(Q_{s,a}).
    
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
        Array of shape (mdp.nS, 1), each V[s] is the value of state s under 
        the reward r and Boltzmann policy.
    2D numpy array
        Array of shape (mdp.nS, mdp.nA), each Q[s,a] is the value of 
        state-action pair [s,a] under the reward r and Boltzmann policy.
    2D numpy array
        Array of shape (mdp.nS, mdp.nA), each value p[s,a] is the probability 
        of taking action a in state s.
    '''
    
    V = np.copy(r)

    t = 0
    diff = float("inf")
    while diff > threshold:
        V_prev = np.copy(V)
        
        # Q[s,a] = (r_s + gamma * \sum_{s'} p(s'|s,a)V_{s'})
        Q = r.reshape((-1,1)) + gamma * np.dot(mdp.T, V_prev)
        # V_s = max_a(Q_sa)
        V = np.amax(Q, axis=1)

        diff = np.amax(abs(V_prev - V))
        
        t+=1
        if horizon is not None:
            if t==horizon: break
    
    V = V.reshape((-1, 1))

    # Compute policy
    # Assigns equal probability to taking actions whose Q_sa == max_a(Q_sa)
    max_Q_index = (Q == np.tile(np.amax(Q,axis=1),(Q.shape[1],1)).T)
    policy = max_Q_index / np.sum(max_Q_index, axis=1).reshape((-1,1))

    return V, Q, policy


def softmax(x, t=1):
    '''
    Numerically stable computation of t*log(\sum_i^n exp(x_i / t))
    '''

    if len(x.shape) == 1: x = x.reshape((1,-1))
    if t == 0: return np.amax(x, axis=1)
    if x.shape[1] == 1: return x
   
    def softmax_2_arg(x1,x2, t):
        ''' 
        Numerically stable computation of log(exp(x1) + exp(x2))
        described in Algorithm 9.2 of Ziebart's PhD thesis 
        http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf.
        '''
        tlog = lambda x: t * np.log(x)
        expt = lambda x: np.exp(x/t)
                
        max_x = np.amax((x1,x2),axis=0)
        min_x = np.amin((x1,x2),axis=0)    
        return max_x + tlog(1+expt((min_x - max_x)))
    
    sm = softmax_2_arg(x[:,0],x[:,1], t)
    # Use the following property of softmax_2_arg:
    # softmax_2_arg(softmax_2_arg(x1,x2),x3) = log(exp(x1) + exp(x2) + exp(x3))
    # which is true since
    # log(exp(log(exp(x1) + exp(x2))) + exp(x3)) = log(exp(x1) + exp(x2) + exp(x3))
    for (i, x_i) in enumerate(x.T):
        if i>1: sm = softmax_2_arg(sm, x_i, t)
    return sm


def mellowmax(x, t=1):
    '''
    Numerically stable computation of t*log(1/n \sum_i^n exp(x_i/t))
    '''
    if len(x.shape) == 1: x = x.reshape((1,-1))
    if t == 0: return np.amax(x, axis=1)
    if x.shape[1] == 1: return x
    
    tlog = lambda x: t * np.log(x)
    expt = lambda x: np.exp(x/t)
    
    def mellowmax_2_arg(x1,x2, i=1):
        ''' 
        Numerically stable computation of 
        log((exp(x1)/i + exp(x2))/(i+1))
        '''
        x1 += -tlog(1/i)
        max_x = np.amax((x1,x2),axis=0)
        min_x = np.amin((x1,x2),axis=0)    
        return max_x + tlog(1+expt((min_x - max_x))) + tlog(1/(i+1))
    
    # Use the following property of softmax_2_arg:
    # softmax_2_arg(softmax_2_arg(x1,x2),x3) = log(exp(x1) + exp(x2) + exp(x3))
    # which is true since
    # log(exp(log(exp(x1) + exp(x2))) + exp(x3)) = log(exp(x1) + exp(x2) + exp(x3))
    mm = mellowmax_2_arg(x[:,0],x[:,1])
    for (i, x_i) in enumerate(x.T):
        if i>1: mm = mellowmax_2_arg(mm, x_i, i)
    return mm
