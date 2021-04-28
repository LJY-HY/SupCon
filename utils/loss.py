import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    """SupConLoss class support both supervised version(SupCon) and unsupervised version(SimCLR)"""
    def __init__(self,temperature = 0.07, base_temperature = 0.07, contrast = True):
        super(SupConLoss,self).__init__()
        self.temperature = temperature
        self.base_temperatrue= base_temperature
        self.contrast = contrast

    def forward(self,features, labels):
        """"Compute Contrastive Loss
        1. Input 
            features    : [bsz*contrast_count, feature_dim]
            labels      : [bsz]
        2. Value derived in the middle
            sim_matrix  : values of dot product between z(i)s
                          [bsz*contrast_count, bsz*contrast_count]
            mask        : imform whether i's label is same as j's label
                          [bsz,bsz]
        3. Output
            loss value  : [1]
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        bsz = labels.shape[0]
        contrast_count = int(features.shape[0]/bsz)

        # Make sim_matrix
        sim_matrix = torch.matmul(features, features.T)
        sim_matrix = torch.div(sim_matrix, self.temperature)
        row_max, max_idx = torch.max(sim_matrix, dim=1, keepdim=True)
        logit_matrix = sim_matrix - row_max.detach()

        # Make mask
        labels = labels.contiguous().view(-1,1)
        mask = torch.eq(labels, labels.T).float()
        mask = mask.repeat(contrast_count,contrast_count)

        # Exclude self-contrast from mask
        self_mask = torch.ones(bsz*contrast_count, bsz*contrast_count, dtype = mask.dtype).scatter(1,torch.arange(bsz * contrast_count).view(-1, 1), 0).to(device)
        mask = mask * self_mask     # element-wise multiplication

        # Compute exp
        exp_matrix = torch.exp(logit_matrix)*self_mask
        log_prob = logit_matrix - torch.log(torch.sum(exp_matrix,1,keepdim=True))
        
        # Compute numerator/denominator
        numerator = (mask*log_prob).sum(1)
        denominator = mask.sum(1)
        if contrast_count == 1:
            denominator[torch.where(denominator<=0)]=1
        loss =-(self.temperature/self.base_temperatrue)*(numerator/denominator)
        loss = loss.mean()
        
        return loss
