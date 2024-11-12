# # tests/test_train.py
# import pytest
# import torch
# from torch.optim import SGD
# from torch import optim
# from tqdm import tqdm
# import sys
# import math
# import pandas as pd
# import numpy as np
# # from train import train_one_epoch  # Importe sua função de treinamento

# def test_train_one_epoch():
#     model = torch.load('./modelo_completo.pt')
#     optimizer = SGD(model.parameters(), lr=0.01)
    
#     dummy_loader = torch.utils.data.DataLoader(
#         [(torch.randn(3, 224, 224), {'boxes': torch.randn(1, 4), 'labels': torch.tensor([1])})],
#         batch_size=1
#     )

#     train_one_epoch(model, optimizer, dummy_loader, torch.device('cuda'), epoch=0)

#     print("Training test passed: one epoch completed")


# def train_one_epoch(model, optimizer, loader, device, epoch):
#     model.to(device)
#     model.train()
    
#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000 # do lr warmup
#         warmup_iters = min(1000, len(loader) - 1)
        
#         lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = warmup_factor, total_iters=warmup_iters)
    
#     all_losses = []
#     all_losses_dict = []
    
#     for images, targets in tqdm(loader):
#         images = list(image.to(device) for image in images)
#         targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
        
#         loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
#         losses = sum(loss for loss in loss_dict.values())
#         loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
#         loss_value = losses.item()
        
#         all_losses.append(loss_value)
#         all_losses_dict.append(loss_dict_append)
        
#         if not math.isfinite(loss_value):
#             print(f"Loss is {loss_value}, stopping trainig") # train if loss becomes infinity
#             print(loss_dict)
#             sys.exit(1)
        
#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()
        
#         if lr_scheduler is not None:
#             lr_scheduler.step() # 
        
#     all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
#     print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
#         epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
#         all_losses_dict['loss_classifier'].mean(),
#         all_losses_dict['loss_box_reg'].mean(),
#         all_losses_dict['loss_rpn_box_reg'].mean(),
#         all_losses_dict['loss_objectness'].mean()
#     ))