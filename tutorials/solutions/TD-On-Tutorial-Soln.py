'''
On-Policy TD Algorithm
'''

import numpy as np
import pandas as pd
import random
import copy

# Very similar to MC GridWorld, but with some changes

class GridWorld:
    
    def __init__(self,max_steps):
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
        """
        Resets environment by setting number of steps taken to zero,
        and sets the beginning state to one of a few options

        Returns: integer representing the current grid square of the agent
        """
        self.num_steps = 0
        self.curr_state = random.choice([0,4,7,9,14])# not exhaustive, on purpose!!
        return self.curr_state

    def pick_action(self,list_probs):
        '''
        list_probs: list of probabilites that add up to 1

        Returns: direction from self.actions
        '''
        length = len(list_probs)
        # cummulative array of probabilites
        cu_list = [sum(list_probs[0:x:1]) for x in range(0, length+1)][1:]
        
        choice = random.random()
        for i in range(length):
            if choice <= cu_list[i]:
                return self.actions[i]
        print('unsafe pick_action')
        return -1

    def take_action(self,action):
        '''
        action: string from env.actions

        Hint: self.actions.index(action) may be helpful

        Returns:
            reward: the reward we get for being in next_state
            next_state: the state that taking action from self.curr_state takes us into. Then update self.curr_state
            done: Bool indicating if self.curr_state is in terminating position or if number steps greater than max_steps
        '''
        possible_next_spaces = self.get_neighbors(self.curr_state)
        action_index = self.actions.index(action)
        next_state = possible_next_spaces[action_index]

        self.curr_state = next_state
        self.num_steps += 1
        reward = self.rewards[next_state]

        done = False
        if self.curr_state == 10 or self.num_steps > self.max_steps:
            done = True

        return reward, next_state, done

###########

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
    q_func, policy: Pandas dataframes
    states_seen: set
    epsilon: number between 0 and 1 exclusive

    Returns: Policy updated in accordance with step (c) in MC-Control-OnPolicy-epsilon-soft.jpg
    '''
    for state in states_seen:
        a_star = q_func.loc[state,:].idxmax()
        for direction in policy.columns:
            if direction == a_star:
                policy.loc[state, direction] = 1 - epsilon + (epsilon/len(policy.columns))
            else:
                policy.loc[state, direction] = (epsilon/len(policy.columns))
    return policy

def on_policy_td_control(env:GridWorld, num_episodes, epsilon, alpha, gamma):
    q_func = pd.DataFrame(0,index=[i for i in range(env.num_states)],columns=env.actions)
    policy = pd.DataFrame(0,index=[i for i in range(env.num_states)],columns=env.actions)
    policy = init_e_soft_policy(policy, epsilon)

    for i in range(num_episodes):
        # init S
        s = env.reset_env()
        # Choose A from S using policy derived from Q
        a = env.pick_action(policy.loc[s,:].values)

        states_seen = set()
        states_seen.add(s)
        done = False
        while not done: # repeat for each step of episode
                            # until S is terminal
            # take action A, observe R, S prime
            reward, s_prime, done = env.take_action(a)
            states_seen.add(s_prime)
            # chose A prime from S prime using policy derived from Q
            a_prime = env.pick_action(policy.loc[s_prime,:].values)
            # update Q(S,A)
            part1 = q_func.loc[s,a]
            part2 = q_func.loc[s_prime,a_prime]
            q_func.loc[s,a] = part1 + alpha*(reward + gamma*part2 - part1)
            # shift s,a pair
            s = s_prime
            a = a_prime
        # end while
        # update policy based on new q_func
        policy = update_policy(q_func,policy,states_seen,epsilon)
    return policy, q_func


if __name__=="__main__":
    max_steps = 100 # HAVE to guranteee episodic tasks
    
    env = GridWorld(max_steps)
    
    num_iterations = 600
    gamma = 0.4
    epsilon = 0.8
    alpha = 0.4
    policy,q_func = on_policy_td_control(env, num_iterations, epsilon, alpha, gamma)

    # NOTE: We have high change to circle back to same place. How will gamma and alpha values effect our convergence?
    # try setting both close to one, then setting one/both below 0.5

    env.grid_print(policy,is_policy=True)
    #print(q_func)
    print('done')