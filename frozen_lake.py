import numpy as np
import sys
from six import StringIO, b
import itertools as it

from gym import utils
import discrete_env

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8_multigoal": [
        "FFFFFFF2",
        "FFFFFFFF",
        "FFFHFFFF",
        "SFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFF1"
    ],
}

SEUQUENTIAL_GOAL_MAPS = {
    "9x9_multigoal": [
        "FFFF1FFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "2FFFSFFF2",
        "FFFFFFFHF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFF1FFFF"
    ] ,
        "8x8_multigoal": [
        "FFFFFFF2",
        "FFFFFFFF",
        "FFFHFFFF",
        "SFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFF1"
    ],
}

class FrozenLakeEnv(discrete_env.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="8x8",is_slippery=True):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        nA = 4
        nS = nrow * ncol

        isd = np.array(int(desc == b'S')).astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col
        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,nrow-1)
            elif a==2: # right
                col = min(col+1,ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a-1)%4, a, (a+1)%4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = float(newletter == b'G')
                                li.append((0.8 if b==a else 0.1, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0, newstate, rew, done))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        return outfile

class FrozenLakeEnvMultigoal(discrete_env.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="8x8_multigoal",is_slippery=True,goal=1):
        assert str(goal) in '12'
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col
        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,nrow-1)
            elif a==2: # right
                col = min(col+1,ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        if goal == 1:
            goalchar = b'1'
        else:
            goalchar = b'2'

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'12GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a-1)%4, a, (a+1)%4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'12GH'
                                rew = float(newletter == goalchar)
                                li.append((0.8 if b==a else 0.1, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'12GH'
                            rew = float(newletter == goalchar)
                            li.append((1.0, newstate, rew, done))

        super(FrozenLakeEnvMultigoal, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        return outfile

class FrozenLakeEnvSequentialMultigoal(discrete_env.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S   : starting point, safe
    F   : frozen surface, safe
    H   : hole, fall to your doom
    1|2 : goal for agent 1 or agent 2

    The episode ends when you reach the all the goals for a corresponding agent.
    You receive a reward of 1 each time you reach a goal, and zero otherwise.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="8x8_multigoal", is_slippery=False, goal=1):
        assert str(goal) in '12'
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = SEUQUENTIAL_GOAL_MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        self.num_goals = num_goals = len(np.where(np.logical_or(desc == b'1', desc == b'2'))[0])

        nA = 9
        nS = (2**num_goals * nrow * ncol) + 1
        self.TERMINAL_STATE = TERMINAL_STATE = nS - 1

        # isd = np.array(desc == b'S').astype('float64').ravel()
        # isd /= isd.sum()

        def goal_states_to_int(goals):
            assert len(goals) == num_goals
            assert np.all([g in [0, 1] for g in goals])
            total = 0
            for i in range(num_goals):
                total += 2**i * goals[i]
            return total

        def to_s(row, col, goals):
            return int(((row*ncol + col) << num_goals) + goal_states_to_int(goals))

        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # left-down
                newcol, newrow = max(col-1,0), min(row+1,nrow-1)
                if newcol == col-1 and newrow == row+1:
                    col, row = newcol, newrow
            elif a==2: # down
                row = min(row+1,nrow-1)
            elif a==3: # right-down
                newcol, newrow = min(col+1,ncol-1), min(row+1,nrow-1)
                if newcol == col+1 and newrow == row+1:
                    col, row = newcol, newrow
            elif a==4: # right
                col = min(col+1,ncol-1)
            elif a==5: # right-up
                newcol, newrow = min(col+1,ncol-1), max(row-1,0)
                if newcol == col+1 and newrow == row-1:
                    col, row = newcol, newrow
            elif a==6: # up
                row = max(row-1,0)
            elif a==7: # left-up
                newcol, newrow = max(col-1,0), max(row-1,0)
                if newcol == col-1 and newrow == row-1:
                    col, row = newcol, newrow
            return (row, col)


        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        isd = np.zeros(nS)
        goal_inds = {}
        goal_count = 0
        for row in range(nrow):
            for col in range(ncol):
                if desc[row, col] == b'S':
                    isd[to_s(row,col,[0]*num_goals)] = 1.0
                if desc[row, col] in b'12':
                    goal_inds[(row, col)] = goal_count
                    goal_count += 1

        isd /= np.sum(isd)


        if goal == 1:
            goalchar = b'1'
        else:
            goalchar = b'2'

        for row in range(nrow):
            for col in range(ncol):
                for goals in map(list, it.product([0, 1], repeat=num_goals)):
                    s = to_s(row, col, goals)
                    for a in range(8):
                        li = P[s][a]
                        letter = desc[row, col]
                        action_penalty = -0.01 if a % 2 == 0 else -0.01 * (2.0**0.5)
                        if is_slippery:
                            for b in [(a-2)%8, a, (a+2)%8]:
                                newrow, newcol = inc(row, col, b)
                                newletter = desc[newrow, newcol]
                                newgoals = goals.copy()
                                goal_already_active = False
                                if newletter in b'12':
                                    goal_already_active = bool(newgoals[goal_inds[(newrow, newcol)]])
                                    newgoals[goal_inds[(newrow, newcol)]] = 1.0
                                newstate = to_s(newrow, newcol, newgoals)
                                rew = float(newletter == goalchar and not goal_already_active) or action_penalty
                                li.append((0.8 if b==a else 0.1, newstate, rew, False))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newletter = desc[newrow, newcol]
                            newgoals = goals.copy()
                            goal_already_active = False
                            if newletter in b'12':
                                goal_already_active = bool(newgoals[goal_inds[(newrow, newcol)]])
                                newgoals[goal_inds[(newrow, newcol)]] = 1.0
                            newstate = to_s(newrow, newcol, newgoals)
                            rew = float(newletter == goalchar and not goal_already_active) or action_penalty
                            li.append((1.0, newstate, rew, False))
            
                    li = P[s][8] #exit action
                    li.append((1.0, TERMINAL_STATE, 0.0, True)) #with probability 1, transition to terminal state and receive reward 0.0. done == True

        for a in range(nA):
            li = P[TERMINAL_STATE][a]
            li.append((1.0, TERMINAL_STATE, 0.0, True)) #with probability 1, transition to terminal state and receive reward 0.0. done == True

        super(FrozenLakeEnvSequentialMultigoal, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        
        pos_info = self.s >> self.num_goals
        row, col = pos_info // self.ncol, pos_info % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        
        if self.s != self.TERMINAL_STATE:
            desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        
        if self.lastaction is not None and self.lastaction != 8:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        elif self.s == self.TERMINAL_STATE:
            outfile.write("  (EXITED)\n")
        else:
            outfile.write("\n")
        
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        return outfile

