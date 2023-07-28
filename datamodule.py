import torch
from torch import nn
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime

from datasets import TrainDataset, ValidDataset, TestDataset

class MovieDataModule(LightningDataModule):
    def __init__(self, config, block, p90):
        super().__init__()
        self.config = config
        self.block = block
        self.p90 = p90
        
    def setup(self, stage):
        path = self.config.movie_lens_path
        data = pd.read_csv(path, sep = ',')#[:10000] #slice for testing
            
        self.user_n = len(pd.unique(data["user"]))
        self.item_n = len(pd.unique(data["item"]))

        data["time_new"] = data["time"].map(datetime.datetime.fromtimestamp)  #actual time info

        user_map = {val : i+1 for i, val in enumerate(pd.unique(data["user"]))}
        user_map_back = {i[1] : i[0] for i in user_map.items()}
        item_map = {val : i+1 for i, val in enumerate(pd.unique(data["item"]))}
        data = data.sort_values(['user', 'time']).reset_index(drop = True) #arrange according to user and increasing time in user 
        data.drop(columns = ['rating'], inplace = True) #rating is irrelevant
        #data["time"] = data["time"] - min(data["time"]) #normalising by min time in the whole dataset #time slice idea in original

        g = data.groupby('user')
        time_ranges = g.apply(lambda x: x.time_new.max() - x.time_new.min())
        time_ranges_days = time_ranges.apply(lambda x: x.days) #time ranges in days for users
        time_diffs = g.apply(lambda x: x.time_new.max() - x.time_new) #diff from each action for user from last action


        #Add the splitting code here 
        #change: remove sequences of length 1 or less here itself

        user_seq_dict = defaultdict(list) #changing to seq per user format
        for i in range(len(data)):
            user_seq_dict[user_map[data.loc[i, "user"]]].append(item_map[data.loc[i, "item"]])

        self.user_seq_train = {}
        self.user_seq_valid = {}
        self.user_seq_test = {}

        #MAKE SPLITS 
        split_num = int(0.15*(len(user_seq_dict)))
        count = 1
        for user in user_seq_dict:
            seq = user_seq_dict[user]
            seq_len = len(user_seq_dict[user])
            time_user = user_map_back[user]
            t = self.config.split_range

            time_diffs_u = time_diffs[time_user].reset_index(drop=True) #changes index from 0 to user specific list of actions count
            
            if count > 2*split_num:

                train_samples_idx = time_diffs_u.index[(time_diffs_u <= datetime.timedelta(days = t))].tolist()

                train_in = [seq[i] for i in range(len(seq)) if i not in train_samples_idx]
                train_out = [seq[i] for i in train_samples_idx]

                self.user_seq_train[user] = [train_in, train_out]
            
            elif count <= split_num:

                val_samples_idx = time_diffs_u.index[(time_diffs_u <= datetime.timedelta(days = t))].tolist()

                val_in = [seq[i] for i in range(len(seq)) if i not in val_samples_idx]
                val_out = [seq[i] for i in val_samples_idx]

                self.user_seq_valid[user] = [val_in, val_out]

            elif count <= 2*split_num:

                bp = round(t/3)
                if self.block == "full":

                    test_samples_idx = time_diffs_u.index[(time_diffs_u <= datetime.timedelta(days = t))].tolist()

                    test_in = [seq[i] for i in range(len(seq)) if i not in test_samples_idx]
                    test_out = [seq[i] for i in test_samples_idx]

                elif self.block == "front":

                    test_window = time_diffs_u.index[(time_diffs_u <= datetime.timedelta(days = t))].tolist()
                    test_select = time_diffs_u.index[(time_diffs_u <= datetime.timedelta(days = t)) & (time_diffs_u > datetime.timedelta(days = 2*bp))].tolist()

                    test_in = [seq[i] for i in range(len(seq)) if i not in test_window]
                    test_out = [seq[i] for i in test_select]

                elif self.block == "mid":
                    test_window = time_diffs_u.index[(time_diffs_u <= datetime.timedelta(days = t))].tolist()
                    test_select = time_diffs_u.index[(time_diffs_u <= datetime.timedelta(days = 2*bp)) & (time_diffs_u > datetime.timedelta(days = bp))].tolist()

                    test_in = [seq[i] for i in range(len(seq)) if i not in test_window]
                    test_out = [seq[i] for i in test_select]

                elif self.block == "back":
                    test_window = time_diffs_u.index[(time_diffs_u <= datetime.timedelta(days = t))].tolist()
                    test_select = time_diffs_u.index[(time_diffs_u <= datetime.timedelta(days = bp))].tolist()

                    test_in = [seq[i] for i in range(len(seq)) if i not in test_window]
                    test_out = [seq[i] for i in test_select]

                elif self.block == "bwol":
                    test_window = time_diffs_u.index[(time_diffs_u <= datetime.timedelta(days = t))].tolist()
                    test_select = time_diffs_u.index[(time_diffs_u <= datetime.timedelta(days = bp)) & (time_diffs_u > datetime.timedelta(days = 1))].tolist()

                    test_in = [seq[i] for i in range(len(seq)) if i not in test_window]
                    test_out = [seq[i] for i in test_select]

                self.user_seq_test[user] = [test_in, test_out]
            
            count += 1

    def train_dataloader(self):
        
        maxlen = self.config.max_seq_len
        path = self.config.data_dir

        self.user_pos_train = []
        for user in self.user_seq_train:
            
            if len(self.user_seq_train[user][0]) > 0:
                pos_seq = self.user_seq_train[user][1]
                if len(pos_seq) > self.config.pos_num:
                    for i in range(self.config.pos_num):
                        self.user_pos_train.append((user, -1))
                else:
                    for pos in pos_seq:
                        self.user_pos_train.append((user, pos))

        print(len(self.user_pos_train))

        train_dataset = TrainDataset(self.user_seq_train, self.user_pos_train, self.item_n, self.config.max_seq_len, self.config.negs_ppos)

        return DataLoader(dataset = train_dataset, batch_size = self.config.batch_size, shuffle = True, num_workers = self.config.n_workers, persistent_workers = True)

    def val_dataloader(self):

        maxlen = self.config.max_seq_len
        path = self.config.data_dir

        self.user_pos_val = []
        for user in self.user_seq_valid:
            
            if len(self.user_seq_valid[user][0]) > 0:
                pos_seq = self.user_seq_valid[user][1]
                for pos in pos_seq:
                    self.user_pos_val.append((user, pos))
        
        print(len(self.user_pos_val))

        val_dataset = ValidDataset(self.user_seq_valid, self.user_pos_val, self.item_n, self.config.max_seq_len)

        return DataLoader(dataset = val_dataset, batch_size = self.config.batch_size, shuffle = False, num_workers = self.config.n_workers, persistent_workers = True)

    def test_dataloader(self):
        maxlen = self.config.max_seq_len
        path = self.config.data_dir

        self.user_pos_test = []

        if self.p90:
            for user in self.user_seq_test:
                
                if len(self.user_seq_test[user][0]) > 0:
                    pos_seq = self.user_seq_test[user][1]
                    self.user_pos_test.append((user, pos_seq[0])) #each user passed once
        else:
            for user in self.user_seq_test:
                
                if len(self.user_seq_test[user][0]) > 0:
                    pos_seq = self.user_seq_test[user][1]
                    for pos in pos_seq:
                        self.user_pos_test.append((user, pos))
        
        print(len(self.user_pos_test))

        test_dataset = TestDataset(self.user_seq_test, self.user_pos_test, self.item_n, self.config.max_seq_len)

        return DataLoader(dataset = test_dataset, batch_size = self.config.batch_size, shuffle = False, num_workers = self.config.n_workers, persistent_workers = True)
