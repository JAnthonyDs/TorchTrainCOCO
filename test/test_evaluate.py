# # tests/test_evaluate.py
# import pytest
# import torch
# from evaluate import evaluate_model  # Importe a função de avaliação

# from torch.utils.data import DataLoader

# def test_evaluate_model():
#     model = torch.load('./modelo_completo.pt')
#     model.eval()  # Coloca o modelo em modo de avaliação
    
#     # Cria um DataLoader para a base de testes (usando dados dummy)
#     dummy_loader = DataLoader(
#         [(torch.randn(3, 224, 224), {'boxes': torch.randn(1, 4), 'labels': torch.tensor([1])})],
#         batch_size=1
#     )
    
#     # Avalia o modelo
#     evaluate_model(model, dummy_loader, torch.device('cpu'))

#     print("Evaluation test passed")


