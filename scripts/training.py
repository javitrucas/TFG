import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn
import os
from model import MILModel
from bag_creator import BagCreator
from graphs import Graphs


class Training:
    def __init__(self, train_bags, train_labels, val_bags, val_labels, num_epochs=10, learning_rate=1e-3, output_model_dir = './models', output_graphs_dir = './output/10_inst/training_graphs'):
        # Inicialización de parámetros
        self.train_bags = train_bags
        self.train_labels = train_labels
        self.val_bags = val_bags
        self.val_labels = val_labels
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Crear el DataLoader para el conjunto de entrenamiento
        self.train_dataset = TensorDataset(torch.stack(train_bags), train_labels)
        self.val_dataset = TensorDataset(torch.stack(val_bags), val_labels)
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)

        # Crear el modelo, optimizador y función de pérdida
        self.model = MILModel()
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()

        # Inicializar listas para almacenar las métricas
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        # Crear el directorios
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

            for bag, label in self.train_dataloader:
                self.optimizer.zero_grad()
                output, _ = self.model(bag[0])  # Desempaquetar solo el 'output', ignorando los 'attention_scores'
                loss = self.criterion(output, label.float().unsqueeze(1))
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

                # Calcular la precisión de entrenamiento
                predicted = (output > 0.5).float()  # umbral de 0.5 para clasificación binaria
                correct_train += (predicted == label.unsqueeze(1)).sum().item()
                total_train += label.size(0)

            self.train_losses.append(total_train_loss)
            train_accuracy = correct_train / total_train
            self.train_accuracies.append(train_accuracy)

            # Evaluación en el conjunto de validación
            self.model.eval()  # Modo evaluación
            total_val_loss = 0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for bag, label in self.val_dataloader:
                    output, _ = self.model(bag[0])  # Desempaquetar solo el 'output', ignorando los 'attention_scores'
                    loss = self.criterion(output, label.float().unsqueeze(1))

                    total_val_loss += loss.item()

                    # Calcular la precisión de validación
                    predicted = (output > 0.5).float()
                    correct_val += (predicted == label.unsqueeze(1)).sum().item()
                    total_val += label.size(0)

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
        graphs = Graphs(self.num_epochs, self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies, self.output_graphs_dir)
        
        # Generar y guardar las gráficas
        graphs.save_plots()

        # Mostrar las gráficas
        graphs.show_plots()


if __name__ == "__main__":
    # Crear instancia de BagCreator para generar los datos
    bag_creator = BagCreator(target_digit=3, num_bags=1000, num_instances=10)

    # Crear las bolsas y obtener las etiquetas
    bags, labels = bag_creator.create_bags()

    # Dividir las bolsas en entrenamiento y validación
    split_idx = int(len(bags) * 0.8)  # Usamos 80% para entrenamiento y 20% para validación
    train_bags, val_bags = bags[:split_idx], bags[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    # Inicializar el objeto de entrenamiento
    trainer = Training(train_bags, train_labels, val_bags, val_labels, num_epochs=10, learning_rate=1e-3, output_model_dir='./models', output_graphs_dir = './output/10_inst/training_graphs')
    
    # Comenzar el entrenamiento
    trainer.train()
