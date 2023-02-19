from rlcard.games.base import Card
import random

LEFT = {'D': 'HJ', 'H': 'DJ', 'C':'SJ', 'S':'CJ'}

ACTION_SPACE = {
    'pass': 0,
    'pick': 1,
    'call-H': 2,
    'call-D': 3,
    'call-S': 4,
    'call-C': 5,
    'HA': 6,
    'HK': 7,
    'HQ': 8,
    'HJ': 9,
    'HT': 10,
    'H9': 11,
    'DA': 12,
    'DK': 13,
    'DQ': 14,
    'DJ': 15,
    'DT': 16,
    'D9': 17,
    'SA': 18,
    'SK': 19,
    'SQ': 20,
    'SJ': 21,
    'ST': 22,
    'S9': 23,
    'CA': 24,
    'CK': 25,
    'CQ': 26,
    'CJ': 27,
    'CT': 28,
    'C9': 29,
    'discard-HA': 30,
    'discard-HK': 31,
    'discard-HQ': 32,
    'discard-HJ': 33,
    'discard-HT': 34,
    'discard-H9': 35,
    'discard-DA': 36,
    'discard-DK': 37,
    'discard-DQ': 38,
    'discard-DJ': 39,
    'discard-DT': 40,
    'discard-D9': 41,
    'discard-SA': 42,
    'discard-SK': 43,
    'discard-SQ': 44,
    'discard-SJ': 45,
    'discard-ST': 46,
    'discard-S9': 47,
    'discard-CA': 48,
    'discard-CK': 49,
    'discard-CQ': 50,
    'discard-CJ': 51,
    'discard-CT': 52,
    'discard-C9': 53
}

ACTION_LIST = list(ACTION_SPACE.keys())

NON_TRUMP = ['9', 'T', 'J', 'Q', 'K', 'A']  # this order matters
SUIT_LIST = ['S', 'H', 'D', 'C']

def init_euchre_deck(customDeck=None):
    ''' Initialize a standard deck of 52 cards
    Parameters:
        customDeck: List of 24 valid euchre cards to be dealt in this order:
                                Across Dealer (5-9)
        Left of Dealer (0-4)                            Right of Dealer(10-14)
                                Dealer (15-19)  Kitty (20-23)
    Returns:
        (list): A list of Card object
    '''
    
    res = [Card(suit, rank) for suit in SUIT_LIST for rank in NON_TRUMP]
    if customDeck is not None:
        result = []
        for card in customDeck:
            if card == 'XX':
                random_card_ind = random.randrange(0,len(res))
                result.append(res[random_card_ind])
                res.pop(random_card_ind)
            else:
                is_valid_card(card)
                result.append(Card(card[0],card[1]))
                res.remove(Card(card[0],card[1]))
        res = result
    return res

def cards2list(cards):
    return [card.get_index() for card in cards]

def is_left(card, trump):
    return card.get_index() == LEFT[trump]

def is_right(card, trump):
    return card.get_index() == trump + 'J'

def is_valid_card(card):
    assert(len(card)==2)
    assert(card[0] in SUIT_LIST)
    assert(card[1] in NON_TRUMP)