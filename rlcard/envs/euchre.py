from rlcard.envs import Env
from rlcard.games.euchre import Game
from rlcard.games.euchre.utils import ACTION_SPACE, ACTION_LIST
from collections import OrderedDict
import numpy as np

class EuchreEnv(Env):

    def __init__(self, config):
        self.game = Game(config=config)
        self.name = "euchre"

        self.actions = ACTION_LIST
        self.state_shape = [213]
        super().__init__(config)


    def _extract_state(self, state):
        """Extract usable information from state."""
        def vec(s):
            # Return 'list' object
            suit = {"C":[1,0,0,0,0], "D":[0,1,0,0,0], "H":[0,0,1,0,0], "S":[0,0,0,1,0], "X":[0,0,0,0,1]}
            rank = {"9":9, "T":10, "J":11, "Q":12, "K":13, "A":14, "X":0}
            if len(s)==1:
                return suit[s[0]]
            temp_list = suit[s[0]]
            temp_list.append(rank[s[1]])
            return temp_list

        state['legal_actions'] = self._get_legal_actions()
        state['raw_legal_actions'] = self.game.get_legal_actions()

        '''
        structure of obs 
        suit = 5-1 Binary Feature | rank = 1 numerical Feature  
        1. Dealer pos relative to Agent:        4-1 Binary Feature                          | 4
        2. suit of trump:                       5-1 Binary Feature                          | 5
        3. Trump caller pos relative to Agent   4-1 Binary Feature                          | 4
        4. Flipped Card                         5-1 Binary Feature and 1 numerical feature  | 6
        5. What happened to the flipped card    3-1 Binary Feature                          | 3
        5a. What card was discarded if Dealer   5-1 Binary Feature and 1 numerical feature  | 6
        6. The led suit for the hand            5-1 Binary Feature                          | 5
        7. Center Cards                     4x  5-1 Binary Feature and 1 numerical feature  | 24
        8. Agents Hand                      6x  5-1 Binary Feature and 1 numerical feature  | 36
        9. Agents Play History              5x  5-1 Binary Feature and 1 numerical feature  | 30
        8. Partners Play History            5x  5-1 Binary Feature and 1 numerical feature  | 30
        9. Left Opponents Play History      5x  5-1 Binary Feature and 1 numerical feature  | 30
        10. Right Opponents Play History    5x  5-1 Binary Feature and 1 numerical feature  | 30
                                                                                    Total:    213

        Notes:
        Perhaps reduce size of player histories by 1 each because end of game redundancy
        Possibly add discraded card to obs structure
        '''

        obs = []
        curr_player_num = state['current_actor']
        # Save which player relative to you is the dealer
        '''1'''
        obs.append( self._orderShuffler(curr_player_num,state['dealer_actor']) )

        '''2 and 3'''
        if state['trump'] is not None:
            obs.append( vec(state['trump']) )
            obs.append( self._orderShuffler(curr_player_num,state['calling_actor']) )
        else: # No Trump called
            obs.append( vec("X") )
            obs.append( [0, 0, 0, 0] )

        '''4'''
        obs.append( vec(state['flipped']) )
        '''5'''
        obs.append( state['flipped_choice'].tolist() )
        '''5a'''
        if state['discarded_card'] is not None and state['dealer_actor'] == curr_player_num:
            obs.append( vec(state['discarded_card']) )
        else:
            obs.append(vec("XX"))
        '''6'''
        if state['lead_suit'] is not None:
            obs.append( vec(state['lead_suit']) )
        else:
            obs.append( vec("X") )

        '''7'''
        # Don't need this because it is already in the history?
        for e in state['center']:
            obs.append(vec(e.get_index()))
        obs.append( (4-len(state['center']))*vec("XX") )
        '''8'''
        for e in state['hand']:
            obs.append(vec(e))
        obs.append( (6-len(state['hand']))*vec("XX") )
        '''
        Need to build 3 hands for each other player
        Note, their hands will grow as mine shrinks
        Because their 'hand' represents which cards I've seen them play
        '''
        '''9 10 11'''
        for i in range(0,4):
            rel_player_num = (i - curr_player_num + 4) % 4
            for e in state['played'][rel_player_num]:
                obs.append(vec(e))
            obs.append( (5-len(state['played'][rel_player_num]))*vec("XX") )

        state['obs'] = np.hstack(obs)

        return state

    def _orderShuffler(self,curr_player_num, player_num):
            '''
            Return encoding of player position relative to curr_player_num.

            As a player, you see the game in this way:
                            Partner = 2
            Left opponent = 1       Right opponent = 3
                            You = 0
            The best players try to decuce which players have which cards as more are revealed.
            It is not so simple as to say "I've seen the King of Spades", you have to remember who threw it.
            Imagine my hand has the Ace of Spades and Ace of Hearts, there are two tricks left, and I lead, and lets say, diamonds are trump.
            I am trying to win the most tricks, which means I want to win THIS hand, if I can, while I lead.
            I want to guess which (if any) off-suits my opponents have. If my partner has thrown off spades before in a previous trick,
            there is a higher chance that they have no more, especially if it was the king of Spades. Thats a powerful signal and it
            tells me that if my partner has any off-suit it probably isn't spades, and if there are any left, the opponents probably have them.
            Playing an off-suit you know your partner doesn't have also gives them the flexibility to trump the first opponents card, or throw off garbage.
            Therefore, since I remembered my partner threw off the King of Spades I deduced they were out of Spades, and thus I threw the Ace of Spades, as
            spades have a higher chance of being in my opponents hands.
            Now consider the above if my Left opponent and Partner had switched hands. See how there is a difference?!

            Also, it's important to remember who was the dealer. As the dealer has an information advantage.
            '''
            bin_encode = [0,0,0,0]
            adjusted_num = (player_num - curr_player_num + 4) % 4
            bin_encode[adjusted_num] = 1
            return bin_encode

    def _decode_action(self, action_id):
        return ACTION_LIST[action_id]

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        legal_ids = {ACTION_SPACE[action]: None for action in legal_actions}
        return OrderedDict(legal_ids)

    def get_payoffs(self):
        return self.game.get_payoffs()