import torch  # Asegúrate de importar torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 128)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class AttentionMechanism(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionMechanism, self).__init__()
        self.V = nn.Linear(input_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1)
    
    def forward(self, h):
        attention_scores = self.w(torch.tanh(self.V(h)))
        weights = F.softmax(attention_scores, dim=0)
        bag_representation = torch.sum(weights * h, dim=0)
        return bag_representation, weights

class MILModel(nn.Module):
    def __init__(self):
        super(MILModel, self).__init__()
        self.feature_extractor = CNNFeatureExtractor()
        self.attention = AttentionMechanism(128, 64)
        self.classifier = nn.Linear(128, 1)
    
    def forward(self, x):
        # Extraer características de cada instancia
        h = [self.feature_extractor(instance.unsqueeze(0)) for instance in x]  # Proceso por instancia
        h = torch.stack(h)  # Convertir la lista de tensores en un solo tensor
        bag_representation, attention_weights = self.attention(h)  # Obtener representación del bag y pesos
        output = torch.sigmoid(self.classifier(bag_representation))  # Predicción final (probabilidad)
        
        return output, attention_weights
        
