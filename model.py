import torch
from torch import nn
import torchmetrics
from pytorch_lightning import LightningModule
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import time

from helper_modules import PointWiseFeedForward, SASRec
from utils import get_nums

class LightSAS(LightningModule):
    def __init__(self, config):
        super().__init__()
        item_num, user_num = get_nums(config)
        self.model = SASRec(item_num, config)
        self.item_num = item_num
        #initialisation apparently makes things faster
        for name, param in self.model.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass # just ignore those failed init layers

        self.config = config
        self.save_hyperparameters()
        self.k_metric = config.k_metric         
        self.loss_fn = nn.BCEWithLogitsLoss() #WRITE
        self.train_state = []
        self.val_state = []
        self.test_state = []
        self.avg_time = 0

    def on_train_epoch_start(self):
        self.time_0 = time.time()

    def training_step(self, batch, batch_idx):
        log_seqs, samples = batch["input_seq"], batch["sample_seq"]

        #print(seqs.get_device())
        #print(seqs.shape)
        ##print("seqs: ", log_seqs)
        ##print("time matrices: ", time_matrices)
        ##print("time matrices: ", time_matrices.shape)
        ##print("pos: ", pos_seqs)
        ##print("neg: ", neg_seqs)

        #print("Starting forward")

        log_feats = self.model(log_seqs)
        final_feat = log_feats[:, -1, :]
        ##print("Forward ended")

        sample_embs = self.model.item_emb(samples).to(self.device) 

        #print("samples embedded: ", samples)

        #sample_logits = (sample_embs * final_feat.unsqueeze(-2)).sum(dim=-1)
        #or (prob faster below)
        sample_logits = (sample_embs @ final_feat.unsqueeze(-1)).squeeze(-1)

        #print("pos score: ", sample_logits[:, 0])
        #print("neg scores: ", sample_logits[:, 1:])

        sample_labels = torch.zeros(sample_logits.shape).to(sample_logits) #device + float for BCE loss
        sample_labels[:, 0] = 1 #changing first col to 1 (only pos sample)

        #print("labels: ", sample_labels)
        #print("labels shape: ", sample_labels.shape)

        #print(sample_logits.reshape(-1))
        #print(sample_labels.reshape(-1))

        loss = self.loss_fn(sample_logits.reshape(-1), sample_labels.reshape(-1))
        
        for param in self.model.item_emb.parameters(): loss += self.config.l2_emb * torch.norm(param)

        self.train_state.append(loss)
        return loss 

    def on_train_epoch_end(self):
        loss = torch.tensor([x.item() for x in self.train_state]).mean()
        self.log('train/loss' , loss, on_step=False, on_epoch=True, prog_bar=True) # on_step = false, on_epoch = true, report average loss over the batch instead of per batch
        self.train_state = []
        elapsed = time.time() - self.time_0
        self.avg_time = self.avg_time + ((elapsed - self.avg_time)/(self.current_epoch+1))

    def validation_step(self, batch, batch_idx):

        log_seqs, item_indices = batch["input_seq"], batch["sample_seq"]

        log_feats = self.model(log_seqs)
        final_feat = log_feats[:, -1, :]  #ONLY THE FINAL EMBEDDING FOR EVAL

        item_embs = self.model.item_emb(item_indices).to(self.device)
        logits = (item_embs @ final_feat.unsqueeze(-1)).squeeze(-1)

        #eval metrics
        logits = -logits  #lower the better here? cause we sorting ascending?
        ranks = logits.argsort().argsort()  #using twice just gets ranks, guessing row-wise
        ranks = [i[0].item() for i in ranks] #positive sample rank for each batch member

        ndcg = 0
        hit = 0
        for rank in ranks:
            if rank < self.k_metric: #when pos rank is in top k then do metric calc 
                ndcg += 1/np.log2(rank + 2)
                hit += 1

        
        ndcg /= log_seqs.shape[0] #considers this as the number of seqs in batch and divides by that
        hit /= log_seqs.shape[0] #same as recall, checked

        self.val_state.append({'ndcg': ndcg, 'hit': hit})
        return {'ndcg': ndcg, 'hit': hit}

    def on_validation_epoch_end(self):
       
        ndcg = torch.tensor([x['ndcg'] for x in self.val_state], dtype = torch.float32).mean()
        self.log('val/ndcg' , ndcg, on_step=False, on_epoch=True, prog_bar=True) # on_step = false, on_epoch = true, report average loss over the batch instead of per batch
        
        hit = torch.tensor([x['hit'] for x in self.val_state], dtype = torch.float32).mean()
        self.log('val/hit' , hit, on_step=False, on_epoch=True, prog_bar=True) # on_step = false, on_epoch = true, report average loss over the batch instead of per batch
        
        self.val_state = []

    def on_test_epoch_start(self):
        self.counts = torch.zeros(10000, device = self.device)
        if (self.config.split_range == 7) or (self.config.split_range == 14):
            self.p90_samples = torch.load("/Users/umergupta/Library/Mobile Documents/com~apple~CloudDocs/Documents/UoE/diss_exps/time/data/P90samples.pt", map_location = 'mps')
        elif (self.config.split_range == 30):
            self.p90_samples = torch.load("/Users/umergupta/Library/Mobile Documents/com~apple~CloudDocs/Documents/UoE/diss_exps/time/data/P90samples.pt", map_location = 'mps')

    def test_step(self, batch, batch_idx):

        log_seqs, item_indices = batch["input_seq"], batch["sample_seq"]

        log_feats = self.model(log_seqs)
        final_feat = log_feats[:, -1, :]  #ONLY THE FINAL EMBEDDING FOR EVAL

        item_embs = self.model.item_emb(item_indices).to(self.device)
        logits = (item_embs @ final_feat.unsqueeze(-1)).squeeze(-1)

        #eval metrics
        logits = -logits  #lower the better here? cause we sorting ascending?
        ranks = logits.argsort().argsort()  #using twice just gets ranks, guessing row-wise
        ranks = [i[0].item() for i in ranks] #positive sample rank for each batch member

        ndcg = 0
        hit = 0
        for rank in ranks:
            if rank < self.k_metric: #when pos rank is in top k then do metric calc 
                ndcg += 1/np.log2(rank + 2)
                hit += 1

        
        ndcg /= log_seqs.shape[0] #considers this as the number of seqs in batch and divides by that
        hit /= log_seqs.shape[0] #same as recall, checked

        #P90 Coverage
        p90_embs = self.model.item_emb(self.p90_samples).to(self.device)
        logits = (final_feat @ p90_embs.t())

        logits = -logits  #lower the better here? cause we sorting ascending?
        ranks = logits.argsort().argsort()  #using twice just gets ranks, guessing row-wise
        in_topk = (ranks < self.k_metric)
        #print(in_topk)
        count_topk = in_topk.sum(0)
        #print(count_topk)
        self.counts += count_topk

        self.test_state.append({'ndcg': ndcg, 'hit': hit})

        return {'ndcg': ndcg, 'hit': hit}

    def on_test_epoch_end(self):
       
        ndcg = torch.tensor([x['ndcg'] for x in self.test_state], dtype = torch.float32).mean()
        self.log('test/ndcg' , ndcg, on_step=False, on_epoch=True, prog_bar=True) # on_step = false, on_epoch = true, report average loss over the batch instead of per batch
        
        hit = torch.tensor([x['hit'] for x in self.test_state], dtype = torch.float32).mean()
        self.log('test/hit' , hit, on_step=False, on_epoch=True, prog_bar=True) # on_step = false, on_epoch = true, report average loss over the batch instead of per batch
        
        self.test_state = []

        #print(self.counts)
        p90_thresh = self.counts.to(torch.float32).quantile(0.9)
        #print(p90_thresh)
        in_p90 = (self.counts > p90_thresh)
        p90 = in_p90.sum()/10000
        self.log('test/p90' , p90, on_step=False, on_epoch=True, prog_bar=True) 

        self.log('epoch_time', self.avg_time, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return Adam(params = self.parameters(), lr=self.config.lr, betas = (0.9, 0.98))  #ADD CODE FOR SCHEDULER ETC


