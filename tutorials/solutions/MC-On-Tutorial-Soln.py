'''
Monte Carlo GridWorld
'''
import numpy as np
import pandas as pd
import random

class GridWorld:
    
    def __init__(self,max_steps,gamma):
        self.terminal_state = 10
        self.blocked_states = [6,11,13]
        self.sz = 4
        self.num_states = self.sz*self.sz
        
        self.possible_actions = 4
        self.actions = ['North','East','South','West']

        self.grid = self.initialize_grid1()
        self.rewards = np.zeros(self.num_states)
        self.rewards[self.terminal_state] = 10 # get 10 reward for getting to terminating state
        ####
        self.max_steps = max_steps
        self.gamma = gamma
        self.reset_env()

    def initialize_grid1(self):
        '''
        Truth of Environment
        Create a szxsz grid as follows
        . . . .     Let . = 0
        . . X .         X = 1
        . . G X         G = 2
        . X . .
        '''
        grid = np.zeros(self.num_states)
        grid[self.terminal_state] = 2
        for b in self.blocked_states:
            grid[b] = 1
        return grid

    def get_neighbors(self,state):
        '''
        Helpful function. Important to understand output
        Returns: list of neighboring states in order [n,e,s,w]. elem in range [0, self.num_states)
        

        If in terminal state or blocked state, you don't transition \n
        If taking a step takes you off the edge, you map back to yourself in that direction \n 
        If taking a step takes you to a blocked state, you map back to yourself in that direction
        '''
        if state == self.terminal_state or state in self.blocked_states:
            return []
        north_n = state - self.sz if state - self.sz >= 0 else state
        east_n = state + 1 if (state + 1) % self.sz > state % self.sz else state
        south_n = state + self.sz if state + self.sz < self.num_states else state
        west_n = state - 1 if (state - 1) % self.sz < state % self.sz else state
        res = [north_n,east_n,south_n,west_n]
        
        for i in range(4):
            if res[i] in self.blocked_states:
                res[i] = state
        return res

    def grid_print(self,input,is_policy=False):
        def tup_2_st(coord:tuple):
            assert(coord[0] >= 0 and coord[0] < self.sz)
            assert(coord[1] >= 0 and coord[1] < self.sz)
            return coord[0]*self.sz + coord[1]
        vals = {0:".",1:"X",2:"G"}
        for r in range(self.sz):
            # print real grid
            for c in range(self.sz):
                print(vals[self.grid[tup_2_st((r,c))]],end=" ")
            print(" | ",end=" ")
            # print your vals
            if is_policy:
                for c in range(self.sz):
                    stuff = {0:"^",1:">",2:"v",3:"<"}
                    next_pos = np.argmax(input.loc[tup_2_st((r,c)),:].values)
                    if vals[self.grid[tup_2_st((r,c))]] == ".":
                        print(stuff[next_pos],end=" ")
                    else:
                        print(vals[self.grid[tup_2_st((r,c))]],end=" ")
            else:
                for c in range(self.sz):
                    print(round(input[r*self.sz+c],2),end=" ")
            print()
        print()
        return

    def reset_env(self):
        self.num_steps = 0
        self.curr_state = random.choice([0,4,7,9,14])# not exhaustive, on purpose!!

    def generate_episode(self,pi):
        '''
        pi: valid policy, like one returned from init_e_soft_policy

        Use pi to navigate through the world.
        Update self.curr_state and self.num_steps

        Returns: list of tuples (state, action, reward)
        '''
        episode = []
        while self.curr_state != self.terminal_state and self.num_steps < self.max_steps:
            # get action according to pi
            action = self._pick_action(pi.loc[self.curr_state,:].values)
            # find next_state based on action
            possible_next_spaces = self.get_neighbors(self.curr_state)
            next_state = possible_next_spaces[action]
            # add tuple to episode
            episode.append((self.curr_state,self.actions[action],self.rewards[next_state]))
            # actually change gridworld with action and increment num_steps
            self.curr_state = next_state
            self.num_steps += 1

        return episode

    def _pick_action(self,list_probs):
        '''
        list_probs: list of probabilites that add up to 1

        Returns: index corresponding to which bin a random number fell into
        '''
        length = len(list_probs)
        # cummulative array of probabilites
        cu_list = [sum(list_probs[0:x:1]) for x in range(0, length+1)][1:]
        
        choice = random.random()
        for i in range(length):
            if choice <= cu_list[i]:
                return i
        print('unsafe pick_action')
        return -1

###########

def get_return(partial_episode,gamma):
    '''
    partial_episode: list of (s_i, a_i, r_i+1) tuples

    Hint: this is like 3 lines

    gamma: discount factor, between 0 and 1
    '''
    returns = 0
    for i in range(len(partial_episode)):
        _,_,this_return = partial_episode[i]
        returns += pow(gamma,i)*this_return
    return returns

def init_e_soft_policy(policy,epsilon):
    '''
    Policy: Pandas dataframe with dimensions (num_states[16], legal actions[4])
    epsilon: number between 0 and 1 exclusive

    for each state
        randomly choose a legal action and assing it probability 1-epsilon + (epsilon / num actions),
        give the other options (epsilon / num actions)

    Returns: Policy as changed by the algorithm described above
    '''
    for ind in policy.index:
        random_choice = random.choice(policy.columns)
        other_choices = [c for c in policy.columns if c != random_choice]
        policy.loc[ind,random_choice] = 1 - epsilon + (epsilon / len(other_choices))
        for o in other_choices:
            policy.loc[ind,o] = epsilon / len(other_choices)
    
    return policy

def update_policy(q_func, policy, states_seen, epsilon):
    '''
    q_func,policy: Pandas dataframes
    states_seen: set
    epsilon: number between 0 and 1 exclusive

    Returns: Policy updated in accordance with step (c) in MC-Control-OnPolicy-epsilon-soft.jpg
    '''
    for state in states_seen:
        a_star = q_func.loc[state,:].idxmax()
        for direction in policy.columns:
            if direction == a_star:
                policy.loc[state,direction] = 1 - epsilon + (epsilon/len(policy.columns))
            else:
                policy.loc[state,direction] = (epsilon/len(policy.columns))
    return policy


def on_policy_first_visit_mc(env:GridWorld,num_iter,epsilon):
    '''
    env: GridWorld object
    num_iter: number >= 1
    epsilon: number between 0 and 1 exclusive

    Returns: policy and q_func
    '''
    # initialize
    q_func = pd.DataFrame(0,index=[i for i in range(env.num_states)],columns=env.actions)
    returns = {(state,action):[] for state in range(env.num_states) for action in env.actions}
    policy = pd.DataFrame(0,index=[i for i in range(env.num_states)],columns=env.actions)
    policy = init_e_soft_policy(policy, epsilon)

    for i in range(num_iter):
        # generate episode using pi
        env.reset_env()
        episode = env.generate_episode(policy)
        # for each s,a pair in episode
        s_a_seen = {}
        states_seen = set()
        for step_ind, step in enumerate(episode):
            state, action, _ = step
            states_seen.add(state)
            if (state,action) not in s_a_seen.keys():
                g = get_return(episode[step_ind:],env.gamma)
                s_a_seen[(state,action)] = g

                returns[(state,action)].append(s_a_seen[(state,action)])
                q_func.loc[state,action] = np.mean(returns[(state,action)])
        # part c
        policy = update_policy(q_func,policy,states_seen,epsilon)
    return policy, q_func

if __name__=="__main__":
    max_steps = 100 # HAVE to guranteee episodic tasks
    gamma = 0.98
    env = GridWorld(max_steps,gamma)
    num_iterations = 500
    epsilon = 0.5
    policy,q_func = on_policy_first_visit_mc(env,num_iterations,epsilon)

    env.grid_print(policy,is_policy=True)
    print('done')