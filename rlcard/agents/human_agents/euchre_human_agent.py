from collections import defaultdict

import numpy as np

from rlcard.games.euchre.utils import LEFT, NON_TRUMP, ACTION_SPACE

class EuchreHumanAgent(object):
    
    def __init__(self,name="Default", mute_state = True):
        #Included for readability in debugger
        self.name = name

        self.mute_state = mute_state
        self.use_raw = False
    
    def step(self,state):
        print("---")
        # Right Now it gives you the option to select any action you want
        legal = state['raw_legal_actions']
        if not self.mute_state:
            print(state)
        else:
            if np.sum(state['flipped_choice']) == 0:
                self._print_dealer(state['dealer_actor'], state['current_actor'])
                print("Center Card:",state['flipped'])
            elif state['flipped_choice'][0] == 1:
                self._print_dealer(state['dealer_actor'], state['current_actor'])
            print("Your Hand:",state['hand'])
            print("Top kitty card: ", state['flipped'])
        

        # The state saves the actual object of the cards in state['center']... not very readable
        played=[]
        for card in state['center']:
            played.append(card.get_index())
        if len(played) != 0:
            print("Center Cards: ",played)
        else:
            print("Your Lead")
        
        #print(f"I am player #{state['current_actor']}")
        print("Your Legal Actions:",legal)
        act = input("Select Legal Action: ")
        while act not in legal:
            act = input("Select Legal Action: ")
        return ACTION_SPACE[act]

    # Below was pasted from euchre_rule_agent

    def _print_dealer(self, dealer_num, curr_player_num):
        """Print if you or someone else is the dealer."""
        relative_num = (curr_player_num - dealer_num + 4) % 4
        print(f"You are player {curr_player_num}, the Dealer is player {dealer_num}")
        if relative_num == 0:
            print("You are the Dealer")
        elif relative_num == 2:
            print("Your Partner is Dealer")
        elif relative_num == 3:
            print("The Player To Your Left is Dealer")
        else:
            print("The Player To Your Right is Dealer")

    # Used in env.py in run(). This is what calls the step function above
    # When you are not training.
    def eval_step(self, state):
        return self.step(state), []