import unittest
import numpy as np

from rlcard.games.euchre.game import EuchreGame as Game


CUSTOM_TEST_DECK = ["H9", "HT", "HJ", "HQ", "HK", 
                   "C9", "CT", "CJ", "CQ", "CK", 
                   "S9", "ST", "SJ", "SQ", "SK", 
                   "D9", "DT", "DJ", "DQ", "DK", 
                   "HA", "CA", "SA", "DA"]

TEST_GAME_CONFIG = {
        'allow_step_back': False,
        'allow_raw_data': False,
        'single_agent_mode' : False,
        'record_action' : False,
        'seed': 14,
        'env_num': 1,
        # Custom Flags
        'custom_deck': CUSTOM_TEST_DECK,
        'custom_dealer_id': 1
        }

class TestEuchreGame(unittest.TestCase):
    
    def test_init_game(self):
        """Test euchre init game."""
        game = Game(config=TEST_GAME_CONFIG)
        self.assertEqual(CUSTOM_TEST_DECK, game.custom_deck)
        self.assertEqual(TEST_GAME_CONFIG.get("custom_dealer_id"), game.custom_dealer)
        game.init_game()
        self.assertEqual(game.flipped_card.get_index(), "HA")
        self.assertEqual(game.current_player, (TEST_GAME_CONFIG.get("custom_dealer_id")+1)%4 )

    def test_playthrough(self):
        """Test playthrough of entire game."""
        game = Game(config=TEST_GAME_CONFIG)
        game.init_game()
        actions = ['pass','pass','pass','pick','discard-D9',
                   'HJ','C9','S9','DJ',
                   'HK', 'CT', 'ST', 'HA',
                   'DK','HQ','CJ','SJ',
                   'HT','CQ','SQ','DT',
                   'H9','CK','SK','DQ']
        
        for action in actions:
            game.step(action)

        self.assertTrue(game.is_over)
        history = [["S9","ST","SJ","SQ","SK"],
                   ["DJ","HA","DK","DT","DQ"],
                   ["HJ","HK","HQ","HT","H9"],
                   ["C9","CT","CJ","CQ","CK"]]
        
        self.assertEqual(game.played,history)
        self.assertEqual(game.score, {0:0, 1:1, 2:4, 3:0})
        payoffs = game.get_payoffs()
        self.assertEqual(payoffs, {0:1, 1:-1, 2:1, 3:-1})


if __name__ == '__main__':
    unittest.main()