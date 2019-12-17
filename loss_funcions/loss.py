import torch
from torch import nn

# Loss functions
disc_criterion = nn.BCEWithLogitsLoss()
# disc_criterion = nn.BCELoss()

def MF1_criterion(result, masks):
    MD1 = torch.mean((result - masks)**2 * masks)
    FA1 = torch.mean((result - masks)**2 * (1 - masks))
    MF_loss1 = MD1*100 + FA1
    return MD1.detach(), FA1.detach(), MF_loss1

def MF2_criterion(result, masks):
    MD2 = torch.mean((result - masks)**2 * masks)
    FA2 = torch.mean((result - masks)**2 * (1 - masks))
    MF_loss2 = MD2 + FA2*1
    return MD2.detach(), FA2.detach(), MF_loss2

adv_loss = nn.BCEWithLogitsLoss()
# adv_loss = nn.BCELoss()

gc_criterion = nn.MSELoss()