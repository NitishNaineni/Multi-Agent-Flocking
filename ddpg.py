from models import Actor, Critic
from experience_replay import Experience


class DDPG:
    def __init__(self, n_obs, n_actions, args):

        self.n_actions = n_actions
        self.n_obs = n_obs


        self.actor = Actor(self.n_obs, self.n_actions)
        self.target_actor = Actor(self.n_obs, self.n_actions)
        
        self.critic = Critic(self.n_obs, self.n_actions)
        self.target_critic = Critic(self.n_obs, self.n_actions)

        # Doing a hard update to make sure the parameters are same

        self.hardUpdate()
        self.hardUpdate()

        # Random Process noise
        self.random_noise = self.OhNoise()

        # Hyper-parameters
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.dist_factor = args.disc
        self.epsilon = 1.0
        

        if args.CUDA: self.cuda_port()

    # Function for updating policy
    def policyUpdate(self):

        # Sampling the batch from experience replay

        # 
        

        return 0


    # Function for hard copying parameters of the main network to target network
    def hardUpdate():

        return 0

    # Function for Soft copying parameters of the main network to target network
    def softUpdate():

        return 0

    # Function for introducing the Noise to the actions
    def OhNoise():

        return 0

