class Config:
    n_agents = 4

    comm_fail_prob = 0
    comm_fail_period = 5

    scheme = None
    csv_filename_prefix = None
    model_filename_prefix = None
    scheme_template = 'maac_{}agents_{}comm'
    experiment_prefix = '../results/'
    csv_filename_prefix_template = '/save/statistics-{}'
    model_filename_prefix_template = '/model/model-{}'
    discrete = False

    random_seed = 2022
    epsilon=0.6
    memory_size = 100000
    batch_size = 32
    grid_width = 20
    grid_height = 20
    fov = 120
    step_size = 2
    xyreso = 1
    yawreso = 5
    sensing_range = 8
    # n_obstacles = 10
    n_targets = 10

    episodes = 300
    max_step = 50
    checkpoint_interval = 50
    update_freq = 250
    gamma = 0.99
    lr_actor = 0.001
    lr_critic = 0.005

    num_options = 3

    hidden_size = 128

    @classmethod
    def update(cls):
        cls.scheme = cls.scheme_template.format(cls.n_agents, cls.comm_fail_prob)
        cls.csv_filename_prefix = cls.csv_filename_prefix_template.format(cls.scheme)
        cls.model_filename_prefix = cls.model_filename_prefix_template.format(cls.scheme)