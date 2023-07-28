import json


class Config():

    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as jsonfile:
            config = json.load(jsonfile)

        self.dir = config['proj_dir']
        self.data_dir = config['data']['dir']
        self.movie_lens_path = config['data']['movie_lens'].replace('$dir', self.data_dir)
        self.checkpoint = config['data']['checkpoint']

        self.batch_size = int(config['hparams']['batch_size'])
        self.num_blocks = int(config['hparams']['num_blocks'])
        self.hidden_dim = int(config['hparams']['hidden_dim'])
        self.split_prop = float(config['hparams']['split_prop'])
        self.split_range = int(config['hparams']['split_range'])
        self.pos_num = int(config['hparams']['pos_num'])
        self.negs_ppos = int(config['hparams']['negs_ppos'])
        self.dropout_rate = float(config['hparams']['dropout_rate'])
        self.l2_emb = float(config['hparams']['l2_emb'])
        self.max_seq_len = int(config['hparams']['max_seq_length'])
        self.num_heads = int(config['hparams']['num_heads'])
        self.k_metric = int(config['hparams']['k_metric'])
        self.epochs = int(config['hparams']['epochs'])
        
        # No of GPUs for training and no of workers for datalaoders
        self.accelerator = config['accelerator']
        self.devices = int(config['devices'])
        self.n_workers = int(config['n_workers'])

        # model checkpoint to continue from
        self.save_dir = config["hparams"]["save_dir"].replace('$proj_dir', self.dir)

        self.run_name = config['run_name']
        
        # LR of optimizer
        self.lr = float(config['hparams']['lr'])
        self.weight_decay = float(config['hparams']['weight_decay'])