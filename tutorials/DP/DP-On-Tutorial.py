'''
Gridworld RL

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
gamma = 0.9

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
def init_transitions():
    q_map = np.zeros((num_states, num_states))
    for i in range(num_states):
        neighbors = [n for n in get_neighbors(i) if n != i]
        for n in neighbors:
            q_map[(i, n)] = 1 / len(neighbors)
    return q_map


'''
TODO Section Below
'''


def policy_evaluation(value_map, policy, rewards, theta):
    '''
    Implements step 2 in algorithm

    Returns: new value_map
    '''
    iterations = 0
    while True:
        improvement = 0
        for s in range(num_states):
            v_s = value_map[s]
            value_map[s] = 0
            for sp in get_neighbors(s):
                if sp != s:
                    prob = policy[s, sp]
                    value_map[s] += prob * (rewards[sp] + gamma * value_map[sp])
            improvement = max(improvement, abs(v_s - value_map[s]))
        grid_print(value_map)
        iterations += 1
        if improvement < theta or iterations > 1000:
            break

    print('iterations', iterations)
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
    rewards = init_reward_map() # Never changes
    theta = 1e-8
    # --
    value_map = init_value_map()
    policy = init_transitions()
    grid_print(policy_evaluation(value_map, policy, rewards, theta))
    print("Policy Iteration: Dynamic Programming")
    

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