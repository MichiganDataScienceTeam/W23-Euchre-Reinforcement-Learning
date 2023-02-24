import random
from copy import copy

from rlcard.games.euchre.utils import cards2list, is_left, is_right, ACTION_SPACE

from rlcard.games.euchre import Dealer
from rlcard.games.euchre import Player
from rlcard.games.euchre import Judger

import numpy as np

class EuchreGame(object):

    def __init__(self, allow_step_back=False,config=None):
        self.allow_step_back = allow_step_back
        self.num_players = 4
        self.payoffs = [0 for _ in range(self.num_players)]

        self.custom_deck = config.get('custom_deck')
        self.custom_dealer = config.get('custom_dealer_id')

    def init_game(self):
        self.payoffs = [0 for _ in range(self.num_players)]

        self.judge = Judger()

        self.dealer = Dealer(custom_deck=self.custom_deck)
        if self.custom_dealer is None:
            self.dealer_player_id = random.randrange(0, self.num_players)
        else:
            self.dealer_player_id = self.custom_dealer
        #print('player',self.dealer_player_id,'is dealer')
        self.players = [Player(i) for i in range(self.num_players)]

        # Deal in order of left, across, right, dealer
        for i in range(self.num_players):
            player = self.players[(i + 1 + self.dealer_player_id) % 4]
            self.dealer.deal_cards(player, 5)

        self.flipped_card = self.dealer.flip_top_card()
        self.calling_player = -1
        # Options: {Avaliable=0,TurnedDown=1,PickedUp=2}
        self.flipped_choice = np.zeros(2)
        self.history = [] # populate with game states
        self.center = [] 
        self.order = []
        self.score = {i:0 for i in range(self.num_players)}
        self.game_over = False
        
        self.trump = None
        self.lead_suit = None
        self.turned_down = None
        self.played = [[] for _ in range(self.num_players)]
        

        self.current_player = self._increment_player(self.dealer_player_id)
        state = self.get_state(self.current_player)
        return state, self.current_player

    def get_state(self, player_id):
        state = {}
        player = self.players[player_id]
        state['hand'] = cards2list(player.hand)
        state['trump_called'] = self.trump is not None
        # Important to remember at each state who called trump
        state['calling_actor'] = self.calling_player
        state['dealer_actor'] = self.dealer_player_id
        state['trump'] = self.trump
        state['turned_down'] = self.turned_down
        state['lead_suit'] = self.lead_suit
        
        state['flipped'] = self.flipped_card.get_index()
        state['flipped_choice'] = self.flipped_choice

        state['center'] = self.center
        state['order'] = self.order
        state['played'] = self.played
        state['current_actor'] = self.current_player
        return state

    def step(self, action):
        if self.allow_step_back:
            self._add_to_history()

        if action == 'pick':
            self._perform_pick_action()
    
        elif action == 'pass':
            self._perform_pass()

        elif action.startswith('call'):
            suit = action.split('-')[1]
            self._perform_call(suit)

        elif action.startswith('discard'):
            card = action.split('-')[1]
            self._perform_discard(card)

        else:
            self._play_card(action)

            if len(self.center) == 4:
                self._end_trick()
                if len(self.players[self.current_player].hand) == 0:
                    self.winner, self.points = self.judge.judge_hand(self)
                    self.game_over = True
        
        state = self.get_state(self.current_player)
        return state, self.current_player

    def _perform_pick_action(self):
        dealer_player = self.players[self.dealer_player_id]
        dealer_player.hand.append(self.flipped_card)
        self.trump = self.flipped_card.suit
        self.flipped_choice[1] = 1
        self.calling_player = self.current_player
        self.current_player = self.dealer_player_id

    def _increment_player(self, player_id):
        return (player_id + 1) % self.num_players

    def _perform_discard(self, card):
        player = self.players[self.current_player]
        for index, hand_card in enumerate(player.hand):
            if hand_card.get_index() == card:
                remove_index = index
                break
        card = player.hand.pop(remove_index)
        self.current_player = self._increment_player(self.current_player)

    def _play_card(self, action):
        player = self.players[self.current_player]
        for index, hand_card in enumerate(player.hand):
            if hand_card.get_index() == action:
                remove_index = index
                break
        card = player.hand.pop(remove_index)
        if len(self.center) == 0:
            if card.suit == self.trump or is_left(card, self.trump):
                self.lead_suit = self.trump
            else:
                self.lead_suit = card.suit
        self.center += [ card ]
        self.played[self.current_player] += [card.get_index()]
        self.order += [ self.current_player ]
        self.current_player = self._increment_player(self.current_player)

    def _end_trick(self):
        winner = self.judge.judge_trick(self)
        self.score[winner] += 1
        self.current_player = winner
        self.center = []
        self.order = []
        self.lead_suit = None

    def _perform_call(self, suit):
        self.trump = suit
        self.current_player = self._increment_player(self.dealer_player_id)

    def _perform_pass(self):
        if self.current_player == self.dealer_player_id:
            self.turned_down = self.flipped_card.suit
            self.flipped_choice[0] = 1
        self.current_player = self._increment_player(self.current_player)

    def get_legal_actions(self):
        hand = self.players[self.current_player].hand
        if len(hand) == 6:
            return [f"discard-{card.get_index()}" for card in hand]

        if self.trump is None:
            if self.turned_down is None:
                return ['pick', 'pass']
            else:
                actions = [f"call-{suit}" for suit in ['S', 'C', 'D', 'H'] if suit != self.turned_down]
                if self.current_player != self.dealer_player_id:
                    actions += ['pass']
                return actions

        if self.lead_suit is None:
            return [card.get_index() for card in hand]

        follow = [card.get_index() for card in hand if 
                    (card.suit == self.lead_suit and not is_left(card, self.trump)) or 
                    (is_left(card, self.lead_suit) and self.lead_suit == self.trump)]

        if len(follow) > 0:
            return follow
        return [card.get_index() for card in hand]

    def get_num_players(self):
        return self.num_players

    def get_payoffs(self):
        payoffs = {}
        
        for i in range(self.num_players):
            if i in self.winner:
                payoffs[i] = self.points
            else:
                payoffs[i] = -self.points

        return payoffs

    def is_over(self):
        return self.game_over

    def get_player_id(self):
        return self.current_player

    def _add_to_history(self):
        '''
        - All Players Hands
        - What card was in the center
        - The avaliability of the center card
        - What is trump
        - Who is the dealer
        - Who is the current player
        - What cards are in the center
        - What card was lead
        - What team has won the last tricks
        - Is the game over
        '''

        return NotImplementedError

    @staticmethod
    def get_num_actions():
        return 54
