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

a = torch.load(model_path)
b = EuchreSimpleRuleAgent()
c = EuchreSimpleRuleAgent()
d = EuchreSimpleRuleAgent()  # always be dealer

agents = [a,b,c,d]
config = {
        'allow_step_back': False, 'allow_raw_data': False, 'single_agent_mode' : False, 'active_player' : 0,
        'record_action' : False, 'seed': None, 'env_num': 1,
        # Custom Flags
        'custom_deck': ["DA","D9","DQ","DT","DJ",
                        "XX","XX","XX","XX","XX",
                        "XX","XX","XX","XX","HJ",
                        "XX","XX","XX","XX","XX",
                        "DK","XX","XX","XX"],
        'custom_dealer_id': 3
        }

num_episodes = 1000

euchre_game = make("euchre", config=config)
euchre_game.set_agents(agents)

actions_taken = {}

for _ in range(num_episodes):
    # setup game
    ext_state, player = euchre_game.reset()
    # make player 1 pass, make player 2 call, dealer discard C9
    #euchre_game.step(ACTION_SPACE["pass"])
    #euchre_game.step(ACTION_SPACE["pick"])
    #next_state, next_player = euchre_game.step(ACTION_SPACE["discard-D9"])
    # observe probabilities for player 1
    taken = a.step(ext_state)
    actions_taken[taken] = actions_taken.get(taken, 0) + 1
    

# print taken action dict
tups = []
for key, val in actions_taken.items():
    tups.append( (ACTION_LIST[key], val/num_episodes*100) )

tups = sorted(tups, key= lambda x: (x[1], x[0]), reverse=True)

for t in tups:
    print(f"{t[0]} taken {t[1]:.2f}%")
