import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import os
from scripts.MNIST.MNISTmodel import MILModel
from scripts.MNIST.MNISTMILDataset import MNISTMILDataset
from scripts.MIL_utils import MIL_collate_fn
from scripts.MNIST.graphs import Graphs

class Training:
    def __init__(self, train_dataset, val_dataset, num_epochs=10, learning_rate=1e-3, output_model_dir='./models', output_graphs_dir='./output/training_graphs', pooling_type='attention'):
        # Inicialización de parámetros
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.pooling_type = pooling_type

        if(pooling_type == 'attention'):
            output_graphs_dir = output_graphs_dir + '/attention'
        elif(pooling_type == 'mean'):
            output_graphs_dir = output_graphs_dir + '/mean'
        elif(pooling_type == 'max'):
            output_graphs_dir = output_graphs_dir + '/max'
        
        # Crear DataLoader con MIL_collate_fn
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, collate_fn=MIL_collate_fn)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, collate_fn=MIL_collate_fn)
        
        # Crear el modelo, optimizador y función de pérdida
        self.model = MILModel(pooling_type)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()  # Pérdida binaria (etiquetas 0/1)
        
        # Inicializar listas para almacenar las métricas
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Crear directorios de salida
        self.output_model_dir = output_model_dir
        os.makedirs(self.output_model_dir, exist_ok=True)
        self.output_graphs_dir = output_graphs_dir
        os.makedirs(self.output_graphs_dir, exist_ok=True)
        
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()  # Modo entrenamiento
            total_train_loss = 0
            correct_train = 0
            total_train = 0
            
            for bag_data, bag_label, inst_labels, adj_mat, mask in self.train_dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass (desempaquetar la salida del modelo)
                output, _ = self.model(bag_data, mask, adj_mat)  # Ignorar los attention_weights
                
                # Asegurar que las formas coincidan
                bag_label = bag_label.unsqueeze(1).float()  # Convertir a (batch_size, 1)
                
                # Calcular la pérdida
                loss = self.criterion(output, bag_label)
                
                # Backward pass y optimización
                loss.backward()
                self.optimizer.step()
                
                # Actualizar métricas de entrenamiento
                total_train_loss += loss.item()
                predicted = (output > 0.5).float()  # Umbral de 0.5 para clasificación binaria
                correct_train += (predicted == bag_label).sum().item()
                total_train += bag_label.size(0)
            
            # Guardar métricas de entrenamiento
            self.train_losses.append(total_train_loss)
            train_accuracy = correct_train / total_train
            self.train_accuracies.append(train_accuracy)
            
            # Evaluación en el conjunto de validación
            self.model.eval()  # Modo evaluación
            total_val_loss = 0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for bag_data, bag_label, inst_labels, adj_mat, mask in self.val_dataloader:
                    # Forward pass (desempaquetar la salida del modelo)
                    output, _ = self.model(bag_data, mask, adj_mat)  # Ignorar los attention_weights
                    
                    # Asegurar que las formas coincidan
                    bag_label = bag_label.unsqueeze(1).float()  # Convertir a (batch_size, 1)
                    
                    # Calcular la pérdida
                    loss = self.criterion(output, bag_label)
                    
                    # Actualizar métricas de validación
                    total_val_loss += loss.item()
                    predicted = (output > 0.5).float()
                    correct_val += (predicted == bag_label).sum().item()
                    total_val += bag_label.size(0)
            
            # Guardar métricas de validación
            self.val_losses.append(total_val_loss)
            val_accuracy = correct_val / total_val
            self.val_accuracies.append(val_accuracy)
            
            # Imprimir resultados por época
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Entrenamiento - Loss: {total_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            print(f"Validación - Loss: {total_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        # Guardar modelo
        model_path = os.path.join(self.output_model_dir, f"model.pth")
        torch.save(self.model.state_dict(), model_path)
        
        # Crear instancia de la clase Graphs para generar las gráficas
        graphs = Graphs(
            num_epochs=self.num_epochs,
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            train_accuracies=self.train_accuracies,
            val_accuracies=self.val_accuracies,
            output_dir=self.output_graphs_dir
        )
        
        # Generar y guardar las gráficas
        graphs.save_plots()
        # Mostrar las gráficas
        graphs.show_plots()

if __name__ == "__main__":
    # Crear instancias de MNISTMILDataset para entrenamiento y validación
    train_dataset = MNISTMILDataset(subset="train", bag_size=10, obj_label=3)
    val_dataset = MNISTMILDataset(subset="test", bag_size=10, obj_label=3)
    
    # Dividir el conjunto de entrenamiento en entrenamiento (80%) y validación (20%)
    train_split_idx = int(len(train_dataset) * 0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_split_idx, len(train_dataset) - train_split_idx])
    
    # Inicializar el objeto de entrenamiento
    trainer = Training(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=10,
        learning_rate=1e-3,
        output_model_dir='./models/new',
        output_graphs_dir='./output/new/training_graphs'
    )
    
    # Comenzar el entrenamiento
    trainer.train()