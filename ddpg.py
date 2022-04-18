from models import Actor, Critic


class DDPG:
    def __init__(self, n_obs, n_actions, args):

        self.n_actions = n_actions
        self.n_obs = n_obs


        self.actor = Actor(self.n_obs, self.n_actions)
        self.target_actor = Actor(self.n_obs, self.n_actions)
        
        self.critic = Critic(self.n_obs, self.n_actions)
        self.target_critic = Critic(self.n_obs, self.n_actions)

        # Doing a hard update to make sure the parameters are same

        hUpdate()
        hUpdate()

        # Random Process noise
        self.random_noise = oh()

        # Hyper-parameters
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.dist_factor = args.disc
        self.epsilon = 1.0
        

        if USE_CUDA: self.cuda()


    def policy_update(self):

