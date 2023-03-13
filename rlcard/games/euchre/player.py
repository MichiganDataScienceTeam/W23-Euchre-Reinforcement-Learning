from .utils import ACTION_SPACE

class EuchrePlayer(object):

    def __init__(self, player_id):
        self.player_id = player_id
        self.hand = []

    def get_player_id(self):
        return self.player_id
    
    def sort_hand(self):
        """Sort euchre hand.
        Ascending order by their action number in ACTION_LIST

        This is for order of the players hand throughout the game to held with consistency
        """
        self.hand = sorted(self.hand, key = lambda x: ACTION_SPACE[x.get_index()])
