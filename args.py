from platform import architecture
import yaml

class parameter_args:

    def __init__(self, fname='config/parameters.yml'):
        with open(fname, 'r') as file:
            args = yaml.safe_load(file)

        self.architecture = args['architecture']
        self.epoch = args['epoch']
        self.timesteps = args['timesteps']
        self.tau = args['tau']
        self.batch_size = args['batch_size']
        self.disc_factor = args['disc_factor']
        self.actor_hn = args['actor_hn']
        self.critic_hn  = args['critic_hn']
        self.lr_actor  = args['lr_actor']
        self.lr_critic = args['lr_critic']
        self.epoch_val  = args['epoch_val']
        self.CUDA  = args['CUDA']
        self.buffer_size_adversary = args['buffer_size_adversary']
        self.buffer_size_agent = args['buffer_size_agent']
        self.mu = args['mu']
        self.theta = args['theta']
        self.max_sigma = args['max_sigma']
        self.min_sigma = args['min_sigma']
        self.decay_steps = args['decay_steps']

