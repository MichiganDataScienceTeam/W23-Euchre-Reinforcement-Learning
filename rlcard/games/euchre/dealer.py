import random

from rlcard.games.euchre.utils import init_euchre_deck


class EuchreDealer(object):

    def __init__(self,custom_deck = None):
        super().__init__()
        
        self.deck = init_euchre_deck(custom_deck)
        if custom_deck is None:
            self.shuffle()
        #self.print_deck()
        self.hand = []

    def shuffle(self):
        random.shuffle(self.deck)

    def deal_cards(self, player, num):
        for _ in range(num):
            card = self.deck.pop(0)
            player.hand.append(card)

    def flip_top_card(self):
        top_card = self.deck.pop(0)
        return top_card

    def print_deck(self):
        i = 0
        for card in self.deck:
            if(i%5==0):
                print()
            print(card.get_index(),end=" ")
            i += 1
        print()