# import rlcard
from rlcard import make
from rlcard import models
from rlcard.models.euchre_rule_models import EuchreSimpleRuleAgent
from rlcard.agents.human_agents.euchre_human_agent import EuchreHumanAgent

a = EuchreSimpleRuleAgent()
b = EuchreSimpleRuleAgent()
c = EuchreSimpleRuleAgent()
d = EuchreSimpleRuleAgent()

agents = [a,b,c,d]
config = {
        'allow_step_back': False,
        'allow_raw_data': False,
        'single_agent_mode' : False,
        'active_player' : 0,
        'record_action' : False,
        'seed': None,
        'env_num': 1,
        # Custom Flags
        'custom_deck': None,
        'custom_dealer_id': None
        }

euchre_game = make("euchre", config=config)
euchre_game.set_agents(agents)

trajectories, payoffs = euchre_game.run(is_training=False)
t = trajectories[0]

print(payoffs)