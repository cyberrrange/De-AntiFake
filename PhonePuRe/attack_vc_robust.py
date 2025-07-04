import torch
import torch.nn as nn
from torch import Tensor
from tqdm import trange
import os
import soundfile as sf



class EmbAttack():
    
    def __init__(self, 
    model: nn.Module, eps: float, n_iters: int,
    eot_attack_size: int=15,
    eot_defense_size: int=15,
    bpda=False
) -> Tensor:
        
        self.model = model
        self.eps = eps
        self.n_iters = n_iters
        self.eot_attack_size = eot_attack_size
        self.eot_defense_size = eot_defense_size
        self.bpda = bpda

        self.criterion = nn.MSELoss()        
        if self.eot_attack_size > 1 or self.eot_defense_size > 1:
            from adaptive_strategies._EOT import EOT
            self.eot_model = EOT(model=model, loss=self.criterion, EOT_size=eot_attack_size, bpda=self.bpda)
        
    def generate(self, vc_tgt: Tensor, adv_tgt: Tensor, vc_tgt_mel_slices, adv_tgt_mel_slices, file_name=None) -> Tensor:
        ptb = torch.zeros_like(vc_tgt).normal_(0, 1).requires_grad_(True)
        #opt = torch.optim.Adam([ptb])
        opt = torch.optim.Adam([ptb], lr=0.1)
        pbar = trange(self.n_iters)

        with torch.no_grad():
            org_emb = self.model.speaker_encoder(vc_tgt, vc_tgt_mel_slices) # speaker_encoder?
            tgt_emb = self.model.speaker_encoder(adv_tgt, adv_tgt_mel_slices)

        for i in pbar:
            adv_inp = vc_tgt + self.eps * ptb.tanh()
            if self.eot_defense_size > 1:
                self.eot_model.EOT_size = self.eot_defense_size
                # self.eot_model.EOT_batch_size = batch_size
                self.eot_model.use_grad = False
                adv_emb, _, _ = self.eot_model(adv_inp, tgt_emb, org_emb, vc_tgt_mel_slices, file_name = file_name)
            else:
                if self.bpda:
                    with torch.no_grad():
                        defensed_adv_inp = self.model.defense(adv_inp, file_name = file_name, ddpm=True)
                    defensed_adv_inp = adv_inp + (defensed_adv_inp - adv_inp).detach()
                    adv_emb = self.model.speaker_encoder(defensed_adv_inp, vc_tgt_mel_slices)
                else:
                    adv_emb = self.model(adv_inp, vc_tgt_mel_slices, file_name = file_name)
            '''compute gradients'''

            if self.eot_attack_size > 1:
                self.eot_model.EOT_size = self.eot_attack_size
                # self.eot_model.EOT_batch_size = batch_size
                self.eot_model.use_grad = True
                _, _, grad = self.eot_model(adv_inp, tgt_emb, org_emb, vc_tgt_mel_slices, file_name = file_name)
                opt.zero_grad()
                ptb.grad = grad
                opt.step()

            else: 
                loss = self.criterion(adv_emb, tgt_emb) - 0.1 * self.criterion(adv_emb, org_emb)
                opt.zero_grad()
                loss.backward()
                #grad = ptb.grad
                #loss.backward()
                opt.step()
            
                    

        return vc_tgt + self.eps * ptb.tanh()