import os
import torch
import wandb
from scripts.model import MILModel

class Training:
    def __init__(self, train_loader, val_loader, num_epochs, learning_rate, output_model_dir, wandb=None):
        """
        Inicializa el objeto de entrenamiento.
        
        Parámetros:
        - train_loader: DataLoader para el conjunto de entrenamiento.
        - val_loader: DataLoader para el conjunto de validación.
        - num_epochs: Número de épocas de entrenamiento.
        - learning_rate: Tasa de aprendizaje.
        - output_model_dir: Directorio donde se guardará el modelo entrenado.
        - wandb: Objeto de wandb para registro de métricas (opcional).
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_model_dir = output_model_dir
        self.wandb = wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        model = MILModel(feature_dim=512).to(self.device)  # Ajusta feature_dim según tus embeddings
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.BCELoss()  # Pérdida binaria

        for epoch in range(self.num_epochs):
            # Entrenamiento
            model.train()
            train_loss, train_correct = 0.0, 0
            for bag_data, bag_label, _, _, mask in self.train_loader:
                bag_data, bag_label, mask = (
                    bag_data.to(self.device),
                    bag_label.to(self.device),
                    mask.to(self.device),
                )
                optimizer.zero_grad()
                output, _ = model(bag_data, mask=mask)

                # Asegurarse de que las formas coincidan
                #print("Output shape:", output.shape)
                #print("Bag label shape:", bag_label.shape)

                # Asegurarse de que output tenga la forma [batch_size, 1] y bag_label también
                loss = criterion(output, bag_label.float().unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_correct += ((output > 0.5).float() == bag_label.float()).sum().item()

            train_loss /= len(self.train_loader)
            train_accuracy = train_correct / len(self.train_loader.dataset)

            # Validación
            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for bag_data, bag_label, _, _, mask in self.val_loader:
                    bag_data, bag_label, mask = (
                        bag_data.to(self.device),
                        bag_label.to(self.device),
                        mask.to(self.device),
                    )
                    output, _ = model(bag_data, mask=mask)
                    loss = criterion(output, bag_label.float().unsqueeze(1))
                    val_loss += loss.item()
                    val_correct += ((output > 0.5).float() == bag_label.float()).sum().item()

            val_loss /= len(self.val_loader)
            val_accuracy = val_correct / len(self.val_loader.dataset)

            # Registrar métricas en wandb
            if self.wandb:
                self.wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                })

            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Guardar el modelo
        os.makedirs(self.output_model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.output_model_dir, 'model.pth'))