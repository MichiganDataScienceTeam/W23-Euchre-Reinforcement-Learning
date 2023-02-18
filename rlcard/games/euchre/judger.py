from rlcard.games.euchre.utils import is_left, is_right, NON_TRUMP

class EuchreJudger(object):

    def __init__(self):
        pass

    def judge_trick(self, game):
        center_cards = game.center
        trump = game.trump
        leader = game.current_player
        player_order = self._get_player_order(leader)
        center_cards = {}
        for i,p in enumerate(player_order):
            center_cards[p] = game.center[i]

        lead_suit = center_cards[leader].suit
        winning_card = center_cards[leader]
        
        if is_right(winning_card, trump):
            return [k for k, v in center_cards.items() if winning_card == v][0]

        for player in player_order[1:]:
            candidate_card = center_cards[player]
            
            if is_right(candidate_card, trump):
                winning_card = candidate_card
                break

            if is_left(winning_card, trump):
                continue

            if is_left(candidate_card, trump):
                winning_card = candidate_card
                continue

            if (candidate_card.suit != trump):
                if candidate_card.suit == lead_suit:
                    if NON_TRUMP.index(candidate_card.rank) > NON_TRUMP.index(winning_card.rank):
                        winning_card = candidate_card
                        continue
            
            if (candidate_card.suit == trump):
                if candidate_card.suit != winning_card.suit:
                    winning_card = candidate_card
                    continue
                if NON_TRUMP.index(candidate_card.rank) > NON_TRUMP.index(winning_card.rank):
                    winning_card = candidate_card
                    continue
        
        return [k for k, v in center_cards.items() if winning_card == v][0]

    def judge_hand(self, game):
        team_1_score = game.score[0] + game.score[2]
        if team_1_score == 5:
            return [0,2], 2
        elif team_1_score >= 3:
            return [0,2], 1
        elif team_1_score == 0:
            return [1,3], 2
        else:
            return [1,3], 1

    def _get_player_order(self, leader):
        return [(i+leader)%4 for i in range(4)]