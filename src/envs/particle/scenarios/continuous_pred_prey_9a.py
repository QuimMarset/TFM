from envs.particle.scenarios.continuous_pred_prey_3a import Scenario as BaseScenario
from envs.particle.core import World, Agent, Landmark


class Scenario(BaseScenario):

    def make_world(self, args=None):
        world = World()

        # set any world properties first
        world.dim_c = 2
        num_good_agents = 3
        num_adversaries = 9
        num_agents = num_adversaries + num_good_agents # deactivate "good" agents
        num_landmarks = 6

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
            agent.action_callback = None if i < (num_agents-num_good_agents) else self.prey_policy
            agent.view_radius = getattr(args, "agent_view_radius", -1)
            print("AGENT VIEW RADIUS set to: {}".format(agent.view_radius))

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False

        # make initial conditions
        self.reset_world(world)
        self.score_function= getattr(args, "score_function", "sum")
        return world


    def _set_entity_attributes(self):
        self.n_entities_obs = 18
        self.obs_entity_feats = 6
        self.n_entities_state = 162
        self.state_entity_feats = 6
        self.n_entities = 18
