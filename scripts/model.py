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
    def __init__(self, input_dim, hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features, mask=None):
        """
        Parámetros:
        - features: Tensor de forma (batch_size, max_bag_size, feature_dim).
        - mask: Tensor de forma (batch_size, max_bag_size) -> Máscara para ignorar instancias rellenas.
        """
        # Calcular los pesos de atención
        attention_scores = self.attention_layer(features)  # (batch_size, max_bag_size, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, max_bag_size)

        # Aplicar máscara si está presente
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask.bool(), float('-inf'))

        # Normalizar los pesos de atención
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, max_bag_size)

        # Calcular la representación del bag como una combinación ponderada de las características
        bag_representation = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch_size, 1, max_bag_size)
            features  # (batch_size, max_bag_size, feature_dim)
        ).squeeze(1)  # (batch_size, feature_dim)

        return bag_representation, attention_weights

class MILModel(nn.Module):
    def __init__(self, feature_dim=128, dropout_prob=0.5, pooling_type='attention', input_feature_dim=None):
        """
        Parámetros:
        - feature_dim: Dimensión de las características finales después de cualquier reducción.
        - dropout_prob: Probabilidad de dropout.
        - pooling_type: Tipo de pooling ('attention', 'mean', 'max').
        - input_feature_dim: Dimensión de las características de entrada (opcional).
        """
        super(MILModel, self).__init__()
        self.pooling_type = pooling_type
        self.feature_extractor = CNNFeatureExtractor() if input_feature_dim is None else None
        self.dropout = nn.Dropout(dropout_prob)

        # Reducción de dimensionalidad si es necesario
        if input_feature_dim is not None and input_feature_dim != feature_dim:
            self.feature_reduction = nn.Linear(input_feature_dim, feature_dim)
        else:
            self.feature_reduction = None

        # Inicializar atención solo si es necesario
        if pooling_type == 'attention':
            self.attention = AttentionMechanism(feature_dim, hidden_dim=64)
        else:
            self.attention = None

        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, bag_data, mask=None, adj_mat=None):
        batch_size, max_bag_size = bag_data.shape[:2]

        # Extraer características si son imágenes crudas
        if bag_data.dim() == 5:  # (batch, bag_size, channels, height, width)
            instances = bag_data.view(-1, *bag_data.shape[2:])  # Aplanar para procesar cada instancia
            features = self.feature_extractor(instances)  # (batch*bag_size, 128)
            features = features.view(batch_size, max_bag_size, -1)  # (batch, bag_size, 128)
        else:
            features = bag_data  # Características pre-extraídas

        # Reducción de dimensionalidad si es necesario
        if self.feature_reduction is not None:
            features = self.feature_reduction(features)

        features = self.dropout(features)

        # Agregación según el tipo de pooling
        if self.pooling_type == 'attention':
            bag_repr, attention_weights = self.attention(features, mask)
        elif self.pooling_type == 'mean':
            if mask is not None:
                mask = mask.unsqueeze(-1).float()
                features = features * mask
                sum_features = features.sum(dim=1)
                count = mask.sum(dim=1)
                bag_repr = sum_features / count.clamp(min=1)  # Evitar división por cero
            else:
                bag_repr = features.mean(dim=1)
            attention_weights = None
        elif self.pooling_type == 'max':
            if mask is not None:
                features = features.masked_fill(~mask.bool().unsqueeze(-1), float('-inf'))
            bag_repr = features.max(dim=1)[0]
            attention_weights = None
        else:
            raise ValueError(f"Pooling type '{self.pooling_type}' no válido")

        # Clasificación final
        output = self.classifier(bag_repr)
        return output, attention_weights