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
        self.state_shape = [173]
        super().__init__(config)
    

    def _extract_state(self, state):
        """Extract usable information from state."""

        def vec(s):
            """Encode list of cards s"""
            # Return 'list' object
            suit = {"C":[1,0,0,0], "D":[0,1,0,0], "H":[0,0,1,0], "S":[0,0,0,1]}
            rank = {"9":9, "T":10, "J":11, "Q":12, "K":13, "A":14}
            if len(s)==1:
                return suit[s[0]]
            temp_list = suit[s[0]]
            temp_list.append(rank[s[1]])
            return temp_list

        state['legal_actions'] = self._get_legal_actions()
        state['raw_legal_actions'] = self.game.get_legal_actions()

        # TODO write obs
        state['obs'] = []
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