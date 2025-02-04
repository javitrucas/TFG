import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        # Capas convolucionales
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 128)  # Ajustado para MNIST (28x28 -> 7x7 después de pooling)
    
    def forward(self, x):
        # Extracción de características usando capas convolucionales
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Reducción de tamaño a la mitad
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Reducción de tamaño a la mitad nuevamente
        x = x.view(x.size(0), -1)  # Aplanar el tensor
        x = F.relu(self.fc(x))  # Capa fully connected
        return x

class AttentionMechanism(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionMechanism, self).__init__()
        # Mecanismo de atención
        self.V = nn.Linear(input_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1)
    
    def forward(self, h, mask=None):
        # Calcular puntuaciones de atención
        attention_scores = self.w(torch.tanh(self.V(h)))  # Puntuaciones sin normalizar
        
        # Aplicar máscara si está presente
        if mask is not None:
            mask = mask.unsqueeze(-1)  # Asegurar que la máscara tenga la misma forma que attention_scores
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))  # Ignorar instancias rellenas
        
        # Normalizar puntuaciones de atención
        weights = F.softmax(attention_scores, dim=1)  # Normalizar sobre las instancias
        
        # Calcular representación del bag
        bag_representation = torch.sum(weights * h, dim=1)  # Agregación ponderada
        return bag_representation, weights.squeeze(-1)  # Eliminar la última dimensión

class MILModel(nn.Module):
    def __init__(self):
        super(MILModel, self).__init__()
        # Componentes del modelo
        self.feature_extractor = CNNFeatureExtractor()  # Extractor de características
        self.attention = AttentionMechanism(128, 64)  # Mecanismo de atención
        self.classifier = nn.Linear(128, 1)  # Clasificador final
    
    def forward(self, bag_data, mask=None, adj_mat=None):
        """
        Parámetros:
        - bag_data: Tensor de forma (batch_size, max_bag_size, 1, 28, 28) -> Datos de entrada (imágenes).
        - mask: Tensor de forma (batch_size, max_bag_size) -> Máscara para ignorar instancias rellenas.
        - adj_mat: Tensor de forma (batch_size, max_bag_size, max_bag_size) -> Matriz de adyacencia (opcional).
        """
        batch_size, max_bag_size = bag_data.size(0), bag_data.size(1)
        
        # Reorganizar las instancias para procesarlas simultáneamente
        instances = bag_data.view(batch_size * max_bag_size, 1, 28, 28)  # (batch_size * max_bag_size, 1, 28, 28)
        features = self.feature_extractor(instances)  # (batch_size * max_bag_size, feature_dim)
        features = features.view(batch_size, max_bag_size, -1)  # (batch_size, max_bag_size, feature_dim)
        
        # Aplicar máscara si está presente
        if mask is not None:
            mask = mask.bool()  # Convertir a booleano para usar en masked_fill
        
        # Obtener representación del bag y pesos de atención
        bag_representation, attention_weights = self.attention(features, mask=mask)
        
        # Clasificación final
        output = torch.sigmoid(self.classifier(bag_representation))  # Probabilidad de clase positiva
        return output, attention_weights