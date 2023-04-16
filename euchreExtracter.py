# import rlcard
import torch
from rlcard import make
from rlcard import models
from rlcard.models.euchre_rule_models import EuchreSimpleRuleAgent
from rlcard.models.euchre_rule_models import EuchreAdvancedRuleAgent
from rlcard.agents.human_agents.euchre_human_agent import EuchreHumanAgent
from rlcard.games.euchre.utils import ACTION_SPACE, ACTION_LIST

model_path = "experiments/euchre_dqn_v2.5_result/model.pth"
device = torch.device("cpu")

d = torch.load(model_path)
b = EuchreSimpleRuleAgent()
c = EuchreSimpleRuleAgent()
a = EuchreSimpleRuleAgent()  # always be dealer

agents = [a, b, c, d]
config1 = {
    'allow_step_back': False, 'allow_raw_data': False, 'single_agent_mode': False, 'active_player': 0,
    'record_action': False, 'seed': None, 'env_num': 1,
    # Custom Flags
    'custom_deck': ["HA", "XX", "XX", "XX", "XX",
                    "DA", "DK", "DQ", "DT", "DJ",
                    "SA", "XX", "XX", "XX", "HJ",
                    "CA", "XX", "XX", "XX", "XX",
                    "D9", "HQ", "HT", "H9"],
    'custom_dealer_id': 3
}
config = {
    'allow_step_back': False, 'allow_raw_data': False, 'single_agent_mode': False, 'active_player': 0,
    'record_action': False, 'seed': None, 'env_num': 1,
    # Custom Flags
    'custom_deck': ["XX", "XX", "XX", "XX", "XX",
                    "XX", "XX", "XX", "XX", "XX",
                    "XX", "XX", "XX", "XX", "XX",
                    "SA", "S9", "SQ", "ST", "SJ",
                    "SK", "HQ", "HT", "HJ"],
    'custom_dealer_id': 3
}

num_episodes = 1000

euchre_game = make("euchre", config=config)
euchre_game.set_agents(agents)

actions_taken = {}
actions_seen = {}
for _ in range(num_episodes):
    # setup game
    # next_state, player = euchre_game.reset()
    euchre_game.reset()
    # make player 1 pass, make player 2 call, dealer discard C9
    # euchre_game.step(ACTION_SPACE["pass"])
    # euchre_game.step(ACTION_SPACE["pick"])
    # next_state, next_player = euchre_game.step(ACTION_SPACE["discard-D9"])
    next_state, next_player = euchre_game.step(ACTION_SPACE["pick"])
    # observe probabilities for player 1
    taken = a.step(next_state)
    actions_taken[taken] = actions_taken.get(taken, 0) + 1
    possible_actions = euchre_game.game.players[0].hand  # RL agent's hand
    for _, card in enumerate(possible_actions):
        actions_seen[card.get_index()] = actions_seen.get(
            card.get_index(), 0) + 1

# print taken action dict
tups = []
for key, val in actions_taken.items():
    # tups.append((ACTION_LIST[key], val/actions_seen[ACTION_LIST[key]]*100))
    tups.append((ACTION_LIST[key], (val/num_episodes)*100))

tups = sorted(tups, key=lambda x: (x[1], x[0]), reverse=True)

for t in tups:
    print(f"{t[0]} taken {t[1]:.2f}%")
