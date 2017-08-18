import numpy as np


def vi_boltzmann(mdp, gamma, r, horizon=None,  temperature=1, 
                            threshold=1e-16):
    '''
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
        Array of shape (mdp.nS, 1), each V[s] is the value of state s under 
        the reward r and Boltzmann policy.
    2D numpy array
        Array of shape (mdp.nS, mdp.nA), each Q[s,a] is the value of 
        state-action pair [s,a] under the reward r and Boltzmann policy.
    '''
    
    V = np.copy(r)
    Q = np.tile(r, (mdp.nA,1)).T

    t = 0
    diff = float("inf")
    while diff > threshold:
        V_prev = np.copy(V)
        for s in range(mdp.nS):
            for a in range(mdp.nA):
                # Q[s,a] = (r_s + gamma * \sum_{s'} p(s'|s,a)V_{s'})
                Q[s,a] = r[s] + gamma * np.dot(mdp.T[s, a, :], V_prev)
            
            # V_s = log(\sum_a exp(Q_sa))    
            V[s] = mellowmax(Q[s,:], temperature)
            
            if np.sum(np.isnan(V[s])) > 0: 
                raise Exception('NaN encountered in VI, t=',t, 's=',s)

        diff = np.amax(abs(V_prev - V))
        
        t+=1
        if horizon is not None:
            if t==horizon: break

    return V.reshape((mdp.nS, 1)), Q


def compute_policy_boltzmann(mdp, V, Q, temperature):
    '''
    Computes the Boltzmann rational policy \pi_{s,a} = exp(Q_{s,a} - V_s).
    
    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    Q : 2D numpy array
        Array of shape (mdp.nS, mdp.nA), each Q[s,a] is the value of 
        state-action pair [s,a].
    V : 1D numpy array
        Value of each of the states of the MDP.

    Returns
    -------
    2D numpy array
        Array of shape (mdp.nS, mdp.nA), each value p[s,a] is the probability 
        of taking action a in state s.
    '''
    
    if temperature<=0:
        raise ValueError('temperature <= 0')

    expt = lambda x: np.exp(x/temperature)
    tlog = lambda x: temperature * np.log(x)

    policy = np.zeros((mdp.nS, mdp.nA))
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            # exp((Q_{s,a} - V_s - temperature*log(nA))/temperature)
            policy[s,a] = expt(Q[s,a] - V[s] - tlog(mdp.nA))
            
    if np.sum(np.isnan(policy)) > 0: 
        raise Exception('NaN encountered in policy')
    
    return policy


def vi_rational(mdp, gamma, r, horizon=None, threshold=1e-16):
    '''
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
        Array of shape (mdp.nS, 1), each V[s] is the value of state s under 
        the reward r and Boltzmann policy.
    2D numpy array
        Array of shape (mdp.nS, mdp.nA), each Q[s,a] is the value of 
        state-action pair [s,a] under the reward r and Boltzmann policy.
    '''
    
    V = np.copy(r)
    Q = np.zeros(mdp.nS, mdp.nA)

    t = 0
    diff = float("inf")
    while diff > threshold:
        V_prev = np.copy(V)
        for s in range(mdp.nS):
            for a in range(mdp.nA):
                # Q[s,a] = (r_s + gamma * \sum_{s'} p(s'|s,a)V_{s'})
                Q[s,a] = r[s] + gamma * np.dot(mdp.T[s, a, :], V_prev)
            
            # V_s = max_a(Q_sa)
            V[s] = np.amax(Q[s,:])

        diff = np.amax(abs(V_prev - V))

        t+=1
        if horizon is not None:
            if t==horizon: break

    return V.reshape((mdp.nS, 1)), Q


def compute_policy_rational(Q):
    '''
    Computes the Boltzmann rational policy \pi_{s,a} = exp(Q_{s,a} - V_s).
    
    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    Q : 2D numpy array
        Array of shape (mdp.nS, mdp.nA), each Q[s,a] is the value of 
        state-action pair [s,a].
    V : 1D numpy array
        Value of each of the states of the MDP.

    Returns
    -------
    2D numpy array
        Array of shape (mdp.nS, mdp.nA), each value p[s,a] is the probability 
        of taking action a in state s.
    '''
    
    policy = np.zeros_like(Q)
    for s in range(Q.shape[0]):
        policy[s,:] = (Q[s,:] == np.amax(Q[s,:])) / np.sum((Q[s,:] == np.amax(Q[s,:])))
    
    return policy


def softmax(x, t=1):
    '''
    Numerically stable computation of log(\sum_i^n exp(x_i))
    '''
    if t == 0: return np.amax(x)
    if x.shape[0] == 1: return x
   
    def softmax_2_arg(x1,x2, t):
        ''' 
        Numerically stable computation of log(exp(x1) + exp(x2))
        described in Algorithm 9.2 of Ziebart's PhD thesis 
        http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf.
        '''
        tlog = lambda x: t * np.log(x)
        expt = lambda x: np.exp(x/t)
                
        max_x = np.amax((x1,x2))
        min_x = np.amin((x1,x2))    
        return max_x + tlog(1+expt((min_x - max_x)))
    
    sm = softmax_2_arg(x[0],x[1], t)
    # Use the following property of softmax_2_arg:
    # softmax_2_arg(softmax_2_arg(x1,x2),x3) = log(exp(x1) + exp(x2) + exp(x3))
    # which is true since
    # log(exp(log(exp(x1) + exp(x2))) + exp(x3)) = log(exp(x1) + exp(x2) + exp(x3))
    for (i, x_i) in enumerate(x):
        if i>1: sm = softmax_2_arg(sm, x_i, t)
    return sm


def mellowmax(x, t=1):
    '''
    Numerically stable computation of log(1/n \sum_i^n exp(x_i))
    '''
    if t == 0: return np.amax(x)
    if x.shape[0] == 1: return x
    
    tlog = lambda x: t * np.log(x)
    expt = lambda x: np.exp(x/t)
    
    def mellowmax_2_arg(x1,x2, i=1):
        ''' 
        Numerically stable computation of 
        log(    (exp(x1)/i + exp(x2))/(i+1)    )
        '''
        x1 += -tlog(1/i)
        max_x = np.amax((x1,x2))
        min_x = np.amin((x1,x2))    
        return max_x + tlog(1+expt((min_x - max_x))) + tlog(1/(i+1))
    
    # Use the following property of softmax_2_arg:
    # softmax_2_arg(softmax_2_arg(x1,x2),x3) = log(exp(x1) + exp(x2) + exp(x3))
    # which is true since
    # log(exp(log(exp(x1) + exp(x2))) + exp(x3)) = log(exp(x1) + exp(x2) + exp(x3))
    mm = mellowmax_2_arg(x[0],x[1])
    for (i, x_i) in enumerate(x):
        if i>1: mm = mellowmax_2_arg(mm, x_i, i)
    return mm
