# tests/test_model.py
import pytest
import torch  # Importe seu modelo aqui

# Testa a inicialização do modelo
def test_model_initialization():
    model = torch.load('./modelo_completo.pt')
    assert model is not None  

# Testa a execução de uma imagem através do modelo
def test_model_forward_pass():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load('./modelo_completo.pt')

    model.to(device)

    model.eval()  # Coloca o modelo em modo de avaliação

    dummy_input = torch.randn(1, 3, 224, 224)
    
    dummy_input = dummy_input.to(device)

    with torch.no_grad():  # Desativa o cálculo de gradientes
        output = model(dummy_input)

    assert output is not None  # Verifica se o modelo gera uma saída
    
