'''
Gridworld RL

'''

import numpy as np

'''
Global Variables
'''
sz = 4
num_states = sz * sz
terminal_state = 10
terminal_reward = 10
blocked_states = [6,11,13]
grid = None
gamma = 0.3

'''
Global Functions
'''

def tup_2_st(coord:tuple):
    assert(coord[0] >= 0 and coord[0] < sz)
    assert(coord[1] >= 0 and coord[1] < sz)
    return coord[0]*sz + coord[1]

def initialize_grid1():
    '''
    Create a (sz,sz) grid as follows
    . . . .     Let . = 0
    . . X .         X = 1
    . . G X         G = 2
    . X . .
    '''
    grid = np.zeros(num_states)
    grid[terminal_state] = 2
    for b in blocked_states:
        grid[b] = 1
    return grid

# Not useful, only for convenience
def grid_print(input,is_policy=False):
    vals = {0:".",1:"X",2:"G"}
    for r in range(sz):
        # print real grid
        for c in range(sz):
            print(vals[grid[tup_2_st((r,c))]],end=" ")
        print(" | ",end=" ")
        # print your vals
        if is_policy:
            for c in range(sz):
                stuff = {0:"^",1:">",2:"v",3:"<"}
                next_pos = np.argmax(input[tup_2_st((r,c)),:])
                neighs = get_neighbors(tup_2_st((r,c)))
                beep = [True if next_pos == n else False for n in neighs]
                if vals[grid[tup_2_st((r,c))]] == ".":
                    print(stuff[np.argmax(beep)],end=" ")
                else:
                    print(vals[grid[tup_2_st((r,c))]],end=" ")
        else:
            for c in range(sz):
                print(round(input[r*sz+c],2),end=" ")
        print()
    print()
    return

def init_value_map():
    v_map = np.zeros(num_states)
    return v_map

def init_reward_map():
    r_map = np.zeros(num_states)
    r_map[terminal_state] = 10 # get 10 reward 
    return r_map

def get_neighbors(state):
    '''
    returns list of neighboring states in order [n,e,s,w]
    
    If in terminal state or blocked state, you don't transition \n
    If taking a step takes you off the edge, you map back to yourself in that direction \n 
    If taking a step takes you to a blocked state, you map back to yourself in that direction
    '''
    if state == terminal_state or state in blocked_states:
        return []
    north_n = state - sz if state - sz >= 0 else state
    east_n = state + 1 if (state + 1) % sz > state % sz else state
    south_n = state + sz if state + sz < num_states else state
    west_n = state - 1 if (state - 1) % sz < state % sz else state
    res = [north_n,east_n,south_n,west_n]
    
    for i in range(4):
        if res[i] in blocked_states:
            res[i] = state
    return res
def init_policy():
    '''
    Create policy of size (sz^2,sz^2) \n
    this maps state -> state transitions \n
    init every direction as equal probability \n
    '''
    q_map = np.zeros((num_states,num_states))
    for i in range(num_states):
        neighbors = get_neighbors(i)
        for n in neighbors:
            q_map[(i,n)] += 0.25
            
    return q_map


'''
TODO Section Below
'''


def policy_evaluation(value_map,theta):
    '''
    Returns: new (1,num_states) value_map

    '''
    # TODO
    return value_map

def policy_improvement(policy,value):
    '''
    Returns: policy, a (num_states,num_states) array containing transition probabilities from state -> state
             stable, bool indicating if policy == policy created by function end
    '''
    stable = True
    # TODO
    return policy, stable

if __name__=="__main__":
    grid = initialize_grid1() # Exists only to support print statements
    rewards = init_reward_map() # Never changes
    theta = 0 # TODO
    # --
    value_map = init_value_map()
    policy = init_policy()
    print("Policy Iteration: Dynamic Programming")
    

    # --
    value_map = init_value_map()
    policy = init_policy()
    print("Policy Iteration: Closed Form")
    
    
    print("done")