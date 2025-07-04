
import torch.nn as nn
import torch

class EOT(nn.Module):

    def __init__(self, model, loss, EOT_size=1, EOT_batch_size=1, use_grad=True, bpda=True):
        super().__init__()
        self.model = model
        self.loss = loss
        self.EOT_size = EOT_size
        self.EOT_batch_size = EOT_batch_size
        self.EOT_num_batches = self.EOT_size //self.EOT_batch_size
        self.use_grad = use_grad
        self.bpda = bpda
    
    # def forward(self, x_batch, tgt_emb, EOT_num_batches=None, EOT_batch_size=None, use_grad=None):
    #     EOT_num_batches = EOT_num_batches if EOT_num_batches else self.EOT_num_batches
    #     EOT_batch_size = EOT_batch_size if EOT_batch_size else self.EOT_batch_size
    def forward(self, x_batch, tgt_emb, org_emb, vc_tgt_mel_slices, EOT_size=None, EOT_batch_size=None, use_grad=None, file_name=None):
        EOT_size = EOT_size if EOT_size else self.EOT_size
        EOT_batch_size = EOT_batch_size if EOT_batch_size else self.EOT_batch_size
        EOT_num_batches = EOT_size // EOT_batch_size
        use_grad = use_grad if use_grad else self.use_grad
        #print("EOT_size: ", EOT_size, "EOT_batch_size: ", EOT_batch_size, "use_grad: ", use_grad)
        if x_batch.ndim == 2:
            x_batch = x_batch.unsqueeze(0)
        n_audios, n_channels, max_len = x_batch.size()
        grad = None
        embeddings = None
        loss = 0
        # decisions = [[]] * n_audios ## wrong, all element shares the same memory
        #decisions = [[] for _ in range(n_audios)]
        for EOT_index in range(EOT_num_batches):
            # if EOT_index == EOT_num_batches - 1:
            #     batch_size = EOT_size % EOT_batch_size
            #     if batch_size == 0: break
            # else: 
            batch_size = EOT_batch_size
            x_batch_repeat = x_batch.repeat(batch_size, 1, 1)
            if use_grad:
                x_batch_repeat.retain_grad()
            tgt_emb_repeat = tgt_emb.repeat(batch_size, 1)
            org_emb_repeat = org_emb.repeat(batch_size, 1)
            #print(tgt_emb.shape, batch_size, x_batch_repeat.shape, x_batch.shape)

            if self.bpda:
                with torch.no_grad():
                    defensed_x_batch = self.model.defense(x_batch_repeat, file_name = file_name, ddpm=True)
                defensed_x_batch = x_batch_repeat + (defensed_x_batch - x_batch_repeat).detach()
                defensed_x_batch = defensed_x_batch.squeeze(0)
                embeddings_EOT = self.model.speaker_encoder(defensed_x_batch, vc_tgt_mel_slices) # embeddings or logits. Just Name it embeddings. (batch_size, n_spks)
            else:
                embeddings_EOT = self.model(x_batch_repeat, vc_tgt_mel_slices, file_name = file_name) # embeddings or logits. Just Name it embeddings. (batch_size, n_spks)
            #decisions_EOT = embeddings_EOT.max(1, keepdim=True)[1]
            # decisions_EOT, embeddings_EOT = self.model.make_decision(x_batch_repeat) # embeddings or logits. Just Name it embeddings. (batch_size, n_spks)
            loss_EOT = self.loss(embeddings_EOT, tgt_emb_repeat) - 0.1 * self.loss(embeddings_EOT, org_emb_repeat)
            if use_grad:
                loss_EOT.backward(torch.ones_like(loss_EOT), retain_graph=True)

            if EOT_index == 0:
                embeddings = embeddings_EOT.view(EOT_batch_size, -1, embeddings_EOT.shape[1]).mean(0)
                loss = loss_EOT.view(EOT_batch_size, -1).mean(0)
                if use_grad:
                    grad = x_batch_repeat.grad.view(EOT_batch_size, -1, n_channels, max_len).mean(0)
                    x_batch_repeat.grad.zero_()
            else:
                embeddings.data += embeddings_EOT.view(EOT_batch_size, -1, embeddings.shape[1]).mean(0)
                loss.data += loss_EOT.view(EOT_batch_size, -1).mean(0)
                if use_grad:
                    grad.data += x_batch_repeat.grad.view(EOT_batch_size, -1, n_channels, max_len).mean(0)
                    x_batch_repeat.grad.zero_()
            
            #decisions_EOT = decisions_EOT.view(EOT_batch_size, -1).detach().cpu().numpy()
            #for ii in range(n_audios):
            #    decisions[ii] += list(decisions_EOT[:, ii])
        
        embeddings = embeddings / EOT_num_batches
        embeddings = embeddings.view(-1, embeddings.shape[1])
        loss = loss / EOT_num_batches
        loss = loss.view(-1)
        if grad is not None:
            grad = grad / EOT_num_batches
            grad = grad.view(n_channels, max_len)
        #print("grad: ", grad.shape, "embeddings: ", embeddings.shape, "loss: ", loss.shape)
        
        return embeddings, loss, grad #, decisions
    