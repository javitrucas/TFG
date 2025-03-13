import torch
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
    
    def forward(self, h, mask=None):
        attention_scores = self.w(torch.tanh(self.V(h)))
        if mask is not None:
            mask = mask.unsqueeze(-1)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(attention_scores, dim=1)
        bag_representation = torch.sum(weights * h, dim=1)
        return bag_representation, weights.squeeze(-1)

class MILModel(nn.Module):
    def __init__(self, pooling_type='attention'):
        super(MILModel, self).__init__()
        self.pooling_type = pooling_type
        self.feature_extractor = CNNFeatureExtractor()
        
        if pooling_type == 'attention':
            self.attention = AttentionMechanism(128, 64)
        else:
            self.attention = None  # No se usa para mean/max
        
        self.classifier = nn.Linear(128, 1)
    
    def forward(self, bag_data, mask=None, adj_mat=None):
        batch_size, max_bag_size, C, H, W = bag_data.size()  # bag_data shape: [batch_size, bag_size, 1, 28, 28]
        # Colapsamos batch_size y bag_size para procesar cada instancia por separado
        instances = bag_data.view(batch_size * max_bag_size, C, H, W)
        features = self.feature_extractor(instances)  # Ahora features tendrá forma: [batch_size * bag_size, feature_dim]
        features = features.view(batch_size, max_bag_size, -1)
        
        if self.pooling_type == 'attention':
            bag_representation, attention_weights = self.attention(features, mask)
        elif self.pooling_type == 'mean':
            if mask is not None:
                mask = mask.unsqueeze(-1).float()
                features = features * mask
                sum_features = features.sum(dim=1)
                count = mask.sum(dim=1)
                bag_representation = sum_features / count
            else:
                bag_representation = features.mean(dim=1)
            # Generamos pesos uniformes
            attention_weights = torch.ones(batch_size, max_bag_size, device=features.device) / max_bag_size
        elif self.pooling_type == 'max':
            if mask is not None:
                mask = mask.unsqueeze(-1).bool()
                features = features.masked_fill(~mask, float('-inf'))
            bag_representation, _ = features.max(dim=1)
            # Generamos un vector one-hot para la instancia con el máximo valor (por norma o similar)
            instance_norms = torch.norm(features, dim=2)  # [batch_size, bag_size]
            max_idx = instance_norms.argmax(dim=1)         # [batch_size]
            attention_weights = torch.zeros(batch_size, max_bag_size, device=features.device)
            for i in range(batch_size):
                attention_weights[i, max_idx[i]] = 1.0
        else:
            raise ValueError("Pooling type no válido")
        
        output = torch.sigmoid(self.classifier(bag_representation))
        return output, attention_weights

