from network import FeedForwardNN


class PPO():
    def __init__(self, env):
        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)

    def learn(self):pass

