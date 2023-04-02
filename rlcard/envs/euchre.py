from rlcard.envs import Env
from rlcard.games.euchre import Game
from rlcard.games.euchre.utils import ACTION_SPACE, ACTION_LIST, LEFT, NON_TRUMP, SUIT_LIST, is_left
from collections import OrderedDict
import numpy as np

class EuchreEnv(Env):

    def __init__(self, config):
        self.game = Game(config=config)
        self.name = "euchre"

        self.actions = ACTION_LIST
        self.state_shape = [253]
        super().__init__(config)
    

    def _extract_state(self, state):
        """Extract usable information from state."""

        def raw_encode(card):
            suit = {"C":[1,0,0,0], "D":[0,1,0,0], "H":[0,0,1,0], "S":[0,0,0,1]}
            rank = {"9":1, "T":2, "J":3, "Q":4, "K":5, "A":6}
            if len(card) == 1:
                return suit[card]
            return suit[card[0]] + [rank[card[1]]]

        # for left bower
        def card_effective_suit(card, trump):
            if trump is None: return card[0]
            return trump if LEFT[trump] == card else card[0]
        
        def encode_card(card, trump, led, turned_down):
            # when we order up in round 1, we take the perspective of the upcard being trump
            # trump is only None when we consider to order up in round 2
            if trump is None:
                # HACK this probably sucks
                return raw_encode(card) + [0, 0] #+ [NON_TRUMP.index(card[1]), 1 if turned_down != card[0] else 0, SUIT_LIST.index(card[0])]
            raw_suit, raw_rank = card[0], card[1]
            effective_suit = card_effective_suit(card, trump)
            is_trump = effective_suit == trump
            is_led = led is None or effective_suit == led
            # this way we distinguish between the two non-led, non-trump cards
            # for consistency, 0 is always S<->D, H<->C
            # suit_mod_2 = (SUIT_LIST.index(trump) + SUIT_LIST.index(effective_suit)) % 2
            # trump-specific rank ordinal encoding: 9 T (J) Q K A LB RB
            # non-trump rank ordinal encoding: 9 T J Q K A
            # suit ordinal encoding: Trump=>2 Led=>1 None=>0

            # can play? (trump is determined)
            # can lead any card (led is none -> all can play)
            # led not None: must follow suit if any card follows suit
            # can_play = True
            # if led is not None and can_follow_suit:
            #     can_play = is_led

            if is_trump:
                rank_ordinal = NON_TRUMP.index(raw_rank)
                if LEFT[trump] == card:
                    # left bower
                    rank_ordinal = len(NON_TRUMP)
                elif card == trump + 'J':
                    # right bower
                    rank_ordinal = len(NON_TRUMP) + 1
                suit_ordinal = 2
                # return raw_encode(card) #+ [rank_ordinal, suit_mod_2, suit_ordinal]
                return raw_encode(card) + [1 if is_trump else 0, 1 if is_led else 0]
            else:
                rank_ordinal = NON_TRUMP.index(raw_rank)
                suit_ordinal = 1 if is_led else 0
                # return raw_encode(card) #+ [rank_ordinal, suit_mod_2, suit_ordinal]
                return raw_encode(card) + [1 if is_trump else 0, 1 if is_led else 0]


        def encode_cards(cards, size, trump, led, turned_down):
            encoding = []
            for card in cards:
                encoding += [1] + list(encode_card(card, trump, led, turned_down))
            for _ in range(size - len(cards)):
                encoding += [0, 0, 0, 0, 0, 0, 0, 0]
            return encoding

        state['legal_actions'] = self._get_legal_actions()
        state['raw_legal_actions'] = self.game.get_legal_actions()

        # state 0: order up, round 1
        # 0a: order up as non-dealer
        # 0b: order up as dealer
        # state 1: order up, round 2
        # 1a: order up, round 2 as non-dealer
        # 1b: screw the dealer
        # state 1.5: add and discard
        # states 2-6: lead_card and play_card

        opponent_cards = []
        partner_cards = []
        if len(state['center']) >= 1:
            opponent_cards.append(state['center'][-1].suit + state['center'][-1].rank)
        if len(state['center']) >= 2:
            partner_cards.append(state['center'][-2].suit + state['center'][-2].rank)
        if len(state['center']) >= 3:
            opponent_cards.append(state['center'][-3].suit + state['center'][-3].rank)

        opponent_played = [y for x in zip(state['played'][(state['current_actor'] + 1) % 4], state['played'][(state['current_actor'] + 3) % 4]) for y in x]
        can_follow_suit = state['lead_suit'] is None or state['lead_suit'] in set(card_effective_suit(c, state['trump']) for c in state['hand'])
        
        def can_play(card, trump, led):
            if led is not None and can_follow_suit:
                return card_effective_suit(card, trump) == led
            return True
        
        def pad_to(_list, length):
            while len(_list) < length:
                _list += [0]
            return _list
        
        # obs structure:
        # [upcard]
        # [eliminated / round 2?]
        # [did my team call trump?]
        # [number of own cards]
        # [own cards]
        # [dealer number (0 means I'm dealer)]
        # [bit field on whether I can play each card]
        # [discarding?]
        # [discarded card] NOT THERE
        # [opponent cards in center]
        # [partner cards in center]
        # [opponent play history]
        # [partner play history]
        # [my play history]

        # where can a card be?
        # played by opponent 0, played by opponent 1, played by partner, played by me, in center, my hand (can play?), flipped, discarded, taken by other person (possibly discarded), unknown
        # 9*24 = 216

        # which stage of the game can we be in?
        # pick/pass, call second suit, discard, lead, play

        state['obs'] = \
            list(encode_card(state['flipped'], state['trump'], None, state['turned_down'])) + [ # 4
            1 if state['turned_down'] is not None else 0, # 1
            1 if state['calling_actor'] % 2 == state['current_actor'] % 2 else 0, # did my team call trump?
            len(state['hand']), # 1
        ] + encode_cards(state['hand'], 6, state['trump'], state['lead_suit'], state['turned_down']) \
          + self._orderShuffler(state['current_actor'], state['dealer_actor']) \
          + pad_to([int(can_play(c, state['trump'], state['lead_suit'])) for c in state['hand']], 6) \
          + [1 if not state['turned_down'] and state['current_actor'] == state['dealer_actor'] else 0] \
          + encode_cards(opponent_cards, 2, state['trump'], state['lead_suit'], state['turned_down']) \
          + encode_cards(partner_cards, 1, state['trump'], state['lead_suit'], state['turned_down']) \
          + encode_cards(opponent_played, 10, state['trump'], None, state['turned_down']) \
          + encode_cards(state['played'][(state['current_actor'] + 2) % 4], 5, state['trump'], None, state['turned_down']) \
          + encode_cards(state['played'][state['current_actor']], 5, state['trump'], None, state['turned_down']) \

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