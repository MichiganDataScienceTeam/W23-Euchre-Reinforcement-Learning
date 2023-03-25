from collections import defaultdict

import numpy as np
from rlcard.envs import make
from rlcard.games.euchre.utils import LEFT, NON_TRUMP, ACTION_SPACE

class EuchreSimpleRuleAgent(object):

    def __init__(self):
        self.use_raw = False

    def step(self, state):
        legal_actions = state['raw_legal_actions']
        hand = state['hand']

        if len(legal_actions) == 1:
            return ACTION_SPACE[legal_actions[0]]

        if len(hand) == 6:
            suit_counts = self.count_suits(hand, include_left=False)
            worst_suit = min(suit_counts, key = suit_counts.get)
            cards = [card for card in hand if card[0] == worst_suit]
            worst_card = [NON_TRUMP.index(card[1]) for card in cards]
            discard = cards[np.argmin(worst_card)]
            return ACTION_SPACE[f'discard-{discard}']

        if not state['trump_called']:
            suit_counts = self.count_suits(hand)
            best_suit = max(suit_counts, key = suit_counts.get)
            if state['turned_down'] is None:
                if suit_counts[state['flipped'][0]] >= 3:
                    return ACTION_SPACE['pick']
                return ACTION_SPACE['pass']
            else:
                if suit_counts[best_suit] >= 3 and best_suit != state['turned_down']:
                    return ACTION_SPACE[f"call-{best_suit}"]
                if 'pass' not in legal_actions:
                    return ACTION_SPACE[np.random.choice(legal_actions)]
                return ACTION_SPACE['pass']
        
        has_right = (state['trump'] + 'J') in legal_actions
        if has_right and len(state['center']) == 0:
            return ACTION_SPACE[state['trump'] + 'J']

        playable_trump = [card for card in legal_actions if card[0] == state['trump']]
        if len(playable_trump) > 0:
            worst_card = [NON_TRUMP.index(card[1]) for card in playable_trump]
            return ACTION_SPACE[playable_trump[np.argmin(worst_card)]]

        aces = [card for card in legal_actions if card[0] != state['trump'] and card[1] == 'A']
        if len(aces) > 0:
            return ACTION_SPACE[aces[0]]
        
        worst_card = [NON_TRUMP.index(card[1]) for card in legal_actions]
        if len(worst_card) > 0:
            return ACTION_SPACE[legal_actions[np.argmin(worst_card)]]
            
        return ACTION_SPACE[np.random.choice(legal_actions)]        


    def eval_step(self, state):
        return self.step(state), []

    @staticmethod
    def count_suits(hand, include_left=True):
        card_count = defaultdict(int)
        for card in hand:
            card_count[card[0]] += 1
            if include_left:
                if card[1] == 'J':
                    card_count[LEFT[card[0]][0]] += 1
        return card_count

class EuchreSimpleRuleModel(object):
    ''' Euchre Simple Rule Agent Model
    '''

    def __init__(self):
        ''' Load pretrained model
        '''
        env = make('euchre')
        rule_agent = EuchreSimpleRuleAgent()
        self.rule_agents = [rule_agent for _ in range(env.num_players)]

    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return self.rule_agents

'''
'''

class EuchreAdvancedRuleAgent(object):
    def __init__(self):
        self.use_raw = False
        self.rank = {"9":9, "T":10, "J":11, "Q":12, "K":13, "A":14}
        self.left_no_trump = False
        self.partner_no_trump = False
        self.right_no_trump = False
    """   
    def step(self, state):
        '''
    Round 0: Calling Trump:
        I am dealer and was given trump: 
            Consider all non-trump, and non-aces for discarding. Pick the lowest card from the least-frequent suit
        I am deciding trump:
            Don't Call if:
                0 or 1 Trump 
            Instant Call if:
                5 or 4 trump of any kind
            Opponent gets card: 

            I get card (I am Dealer):

            Partner gets card:
    Round 1: Calling Trump

    Round 2: Playing Card 1
            
    Round 3: Playing Card 2
            
    Round 4: Playing Card 3

    Round 5: Playing Card 4

    Round 6: Playing Card 5
        
        '''
        legal_actions = state['raw_legal_actions']
        hand = state['hand']
        # no strategy needed
        if len(legal_actions) == 1:
            return ACTION_SPACE[legal_actions[0]]
        
        # less strategy needed: following suit
        # ALWAYS win the trick
        
        # Figure out which card to discard
        if len(hand) == 6:
            return self._discard_card(hand)
        
        # Pick or pass, rounds 0 and 1
        if not state['trump_called']:
            suit_counts = self.count_suits(hand)
            if state['turned_down'] is None:# round 0
                temp_trump = state['flipped'][0]
                has_right = (temp_trump + 'J' in hand)
                has_left = (LEFT[temp_trump] in hand)
                pass
            else: # round 1
                best_suit = max(suit_counts, key = suit_counts.get)

                if 'pass' not in legal_actions:# stick the dealer
                    return ACTION_SPACE[np.random.choice(legal_actions)]
            return ACTION_SPACE['pass']
        # rest of the rounds below
        
        if len(state['center']) == 0: # I get to lead
            pass
        else: # someone else led
            led_suit = state['lead_suit']
            must_play = [c for c in hand if c[0] == led_suit]
            pass
        
        # Just as a catch-all for missing items
        return ACTION_SPACE[np.random.choice(legal_actions)] 
    """

    def step(self, state):
        """
        Decide how to play based on current state.

        Round 0:
            If all 4 bowers, pass
            Call if 3+ Trump in hand, and not ordering up 1, 2, or 3 to opponent
        Round 1:
            If all 4 bowers, pass
            Call 
        Round 2:
            If opponent ordered up trump, and I have only 1, and aces, play trump
            If partner ordered up trump, don't lead trump
        Round 3-6:

        """
        legal_actions = state['raw_legal_actions']
        hand, flipped = state['hand'], state['flipped']
        my_num, dealer_num = state['current_actor'], state['dealer_actor']
        dealer_rel_pos = self.pos_decode(my_num, dealer_num)
        # no strategy needed
        if len(legal_actions) == 1:
            return ACTION_SPACE[legal_actions[0]]
        # discard
        if len(hand) == 6:
            return self._discard_card(hand)
        
        # round 0
        if state['flipped_choice'][0] == 1:
            return self.round_zero(hand, flipped, dealer_rel_pos)
        # round 1
        elif not state['trump_called']:
            return self.round_one(hand, flipped, dealer_rel_pos)
        # rounds 2-6
        return self.random_action(legal_actions)

    def round_zero(self, hand, flipped, dealer_rel_pos):
        """Decide whether to pick or pass."""
        pot_trump = flipped[0]
        trump_in_hand = [c for c in hand if c[0] == pot_trump or LEFT[pot_trump] == c]
        if len(trump_in_hand) < 2:
            return ACTION_SPACE['pass']
        
        top_three = [pot_trump+"J", LEFT[pot_trump], pot_trump+"A"]
        strong_up_card = (flipped in top_three)
        my_team_picks = (dealer_rel_pos % 2 == 0)

        if len(trump_in_hand) > 2:
            #  partner or me     |     they don't get good card
            if my_team_picks or not strong_up_card:
                return ACTION_SPACE['pick']
        if len(trump_in_hand) == 2 and top_three[0] in hand:
            off_aces = [c for c in hand if c[1] == "A" and c != top_three[2]]
            if len(off_aces) > 1:
                return ACTION_SPACE['pick']

        return ACTION_SPACE['pass']

    def round_one(self, hand, flipped, dealer_rel_pos):
        """Decide whether to call or pass."""
        suits_ranked = self.count_suits(hand, include_left=True, consider_rank=True)
        suits_ranked = sorted(suits_ranked.items(), key = lambda x: (x[1], x[0]), reverse=True)
        not_trump = flipped[0]
        for suit, val in enumerate(suits_ranked):
            if suit == not_trump:
                continue
            off_aces = len([c for c in hand if c[1] == "A" and c[0] != suit])
            # I am dealer or have great hand
            if (dealer_rel_pos == 0 or 
               (val > 33) or
               (off_aces > 0 and val > 29)):
                return ACTION_SPACE[f"call-{suit}"]
                
            # if I am not dealer, and my best able-to-call suit isn't great
            break
        return ACTION_SPACE['pass']

    def eval_step(self, state):
        return self.step(state), []

    def _discard_card(self, hand):
        """Discard card logic."""
        trump = hand[5][0]
        bad_cards = [c for c in hand if c[0] != trump and c[1] != 'A' and c != LEFT[trump]]
        bad_suit_counts = self.count_suits(bad_cards, include_left=False)
        worst_suit = min(bad_suit_counts, key = bad_suit_counts.get)
        worst_cards = [w for w in bad_cards if w[0] == worst_suit]
        worst_cards_vals = [self.rank[w[1]] for w in worst_cards]
        worst_card = bad_cards[np.argmin(worst_cards_vals)]
        return ACTION_SPACE[f'discard-{worst_card}']

    @staticmethod
    def count_suits(hand, include_left=True, consider_rank=False) -> dict:
        rank = {"9":2, "T":3, "J":17, "Q":11, "K":12, "A":15}
        card_count = defaultdict(int)
        for card in hand:
            if consider_rank:
                card_count[card[0]] += rank[card[1]]
            else:
                card_count[card[0]] += 1
            if include_left and card[1] == 'J':
                if consider_rank:
                    card_count[LEFT[card[0]][0]] += 16
                else:
                    card_count[LEFT[card[0]][0]] += 1
        return card_count
    
    @staticmethod
    def random_action(acts):
        return ACTION_SPACE[np.random.choice(acts)]
    
    @staticmethod
    def pos_decode(my_pos, player_pos):
        return (player_pos - my_pos + 4) % 4