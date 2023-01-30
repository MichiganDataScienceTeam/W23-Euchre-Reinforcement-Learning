'''
Gridworld RL

Problem:
Try and find the optimal policy to navigate towards
the goal state:
    . . . .
    . . X .
    . . G X
    . X . .

The locations as numbers are:
    0  1  2  3
    4  5  6  7
    8  9  10 11
    12 13 14 15

TIP:
    look in the init_reward_model and init_transition_matrix
    functions for how to index into each of the respective dataframes
    (use the loc func as well)
'''

import numpy as np
import pandas as pd

'''
Global Variables
'''
sz = 4
num_states = sz * sz
actions = ["North","East","South","West"]
num_actions = len(actions)
terminal_state = 10
terminal_reward = 10
blocked_states = [6,11,13]
non_terminal_states = [0,1,2,3,4,5,7,8,9,12,14,15]
grid = None
gamma = 0.8

'''
Global Functions - Do not Touch
'''

def tup_2_st(coord:tuple):
    """Turn coordinate tuple to location on number line.
        E.g. (3,4) in a 5x5 grid = 3*5 + 4 = 19
    """
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
                next_pos = np.argmax(input.loc[tup_2_st((r,c)),:].values)
                if vals[grid[tup_2_st((r,c))]] == ".":
                    print(stuff[next_pos],end=" ")
                else:
                    print(vals[grid[tup_2_st((r,c))]],end=" ")
        else:
            for c in range(sz):
                print(round(input[r*sz+c],2),end=" ")
        print()
    print()
    return


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


def init_value_map():
    """
    v_map: list of every possible state to fill with value

    Note: this includes non-terminal states, but there values should always
    be zero.
    """
    v_map = np.zeros(num_states)
    return v_map


def init_reward_map():
    """
    Rewards are a function of a (state, action) tuple  

    reward map: Pandas dataframe with dims: (num_states[16], legal actions[4])

    Note: this includes non-terminal states, but there values should always
    be zero.
    """
    reward_map = pd.DataFrame(0, 
                              index=non_terminal_states, 
                              columns=actions)
    reward_map.loc[9, "East"] = 10
    reward_map.loc[14, "North"] = 10
    return reward_map


def init_policy(value):
    '''
    Create policy of size (num states, num actions)

    this holds state -> action transitions  

    Note: this includes non-terminal states, but there values should always
    be zero.
    '''
    policy = pd.DataFrame(value, 
                          index=range(num_states), 
                          columns=actions)
 
    return policy


def init_random_policy():
    """Give equal probability to each state"""
    return init_policy(1/num_actions)


def init_transition_dynamics():
    """
    Create matrix of (s,a) -> s_prime transitions
    rows: (s,a) pair     columns: states
    This will be 1 for all legal actions, and zero if you can't go there

    Note: this includes non-terminal states, but there values should always
    be zero.
    """
    # fill 0 for every (s,a) -> s_prime transition, even illegal moves
    all_pairs = [(s, a) for s in range(num_states) for a in actions]
    transition_matrix = pd.DataFrame(0, 
                              index=pd.MultiIndex.from_tuples(all_pairs),
                              columns=range(num_states))

    # fill transition matrix with 1 if you can go there.
    for s in non_terminal_states:
        neighbors = get_neighbors(s)
        for i in range(4): # for each action
            # prob to get to new position from (s,a) tuple = 1
            transition_matrix[neighbors[i]].loc[(s,actions[i])] = 1

    return transition_matrix

'''
TODO Section Below
'''

def policy_evaluation(value_map, policy, rewards_model, transition_matrix, theta):
    '''
    Implements step 2 in algorithm

    Returns: new value_map
    '''
    while True:
        delta = 0

        for s in non_terminal_states:
            # temp store value of V(s)
            
            # calculate new value for V(s)
            
            # save that value
            
            # change delta
            pass
        
        if delta < theta:
            break

    return value_map


def policy_improvement(value_map, policy, rewards_model, transition_matrix):
    '''
    Returns: policy, a (num_states,num_states) array containing transition probabilities from state -> state
             stable, bool indicating if policy == policy created by function end
    Instead of comparing if a =/= pi(s). run through the loop and do:
        if not policy.equals(new_policy)
    '''
    stable = True
    new_policy = init_policy(0)

    for s in non_terminal_states:
        # skip a = pi(s) step
        # find argmax over all 4 actions to take when in s
        
        # save that action in the new_policy
        pass

    # compare whole new_policy to old policy and set stable = False if not the same
    
    return new_policy, stable


if __name__=="__main__":
    grid = initialize_grid1() # Exists only to support print statements
    rewards_model = init_reward_map() # Never changes
    transition_matrix = init_transition_dynamics() # Never changes
    theta = 1e-4

    # --
    value_map = init_value_map()
    policy = init_random_policy()
    print("Policy Iteration: Dynamic Programming")
    print("Initial Value Map")
    grid_print(value_map)
    print("Initial Policy")
    grid_print(policy, True)

    # --
    # Begin algorithm here
    num_steps = 0

    #--
    print("Final Value Map")
    grid_print(value_map)
    print("Final Policy Map")
    grid_print(policy, True)
    print(f"This problem took {num_steps} iterations to converge.")
    print("done")