import os
import torch
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
from scripts.model import MILModel 

class Training:
    def __init__(self, train_loader, val_loader, num_epochs, learning_rate, output_model_dir, patience=5, wandb=None):
        """
        Inicializa el objeto de entrenamiento.
        
        Parámetros:
        - train_loader: DataLoader para el conjunto de entrenamiento.
        - val_loader: DataLoader para el conjunto de validación.
        - num_epochs: Número de épocas de entrenamiento.
        - learning_rate: Tasa de aprendizaje.
        - output_model_dir: Directorio donde se guardará el modelo entrenado.
        - patience: Número de épocas para early stopping.
        - wandb: Objeto de wandb para registro de métricas (opcional).
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_model_dir = output_model_dir
        self.patience = patience
        self.wandb = wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler('cuda')  # Actualizado para autocasting moderno

    def train(self):
        model = MILModel(feature_dim=512).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()  # Cambiado a BCEWithLogitsLoss

        best_val_auc = 0.0
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            train_loss, train_accuracy = self._train_epoch(model, optimizer, criterion)
            val_loss, val_accuracy, val_metrics = self._validate_epoch(model, criterion)

            # Registrar métricas en wandb
            if self.wandb:
                self.wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    **val_metrics
                })

            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Early stopping basado en AUC
            current_val_auc = val_metrics["val_auc"]
            if current_val_auc > best_val_auc:
                best_val_auc = current_val_auc
                epochs_no_improve = 0
                self._save_final_model(model)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print("Early stopping triggered.")
                    break

    def _train_epoch(self, model, optimizer, criterion):
        model.train()
        train_loss, train_correct = 0.0, 0
        for bag_data, bag_label, _, _, mask in self.train_loader:
            bag_data, bag_label, mask = (
                bag_data.to(self.device),
                bag_label.to(self.device),
                mask.to(self.device),
            )
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):  # Actualizado para autocasting moderno
                output, _ = model(bag_data, mask=mask)
                loss = criterion(output, bag_label.float().unsqueeze(1))

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            train_loss += loss.item()
            train_correct += ((torch.sigmoid(output) > 0.5).float() == bag_label.float()).sum().item()

        train_loss /= len(self.train_loader)
        train_accuracy = train_correct / len(self.train_loader.dataset)
        return train_loss, train_accuracy

    def _validate_epoch(self, model, criterion):
        model.eval()
        val_loss, val_correct = 0.0, 0
        all_labels, all_probs = [], []

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
                val_correct += ((torch.sigmoid(output) > 0.5).float() == bag_label.float()).sum().item()

                all_labels.extend(bag_label.cpu().numpy())
                all_probs.extend(torch.sigmoid(output).squeeze(-1).cpu().numpy())

        val_loss /= len(self.val_loader)
        val_accuracy = val_correct / len(self.val_loader.dataset)

        # Calcular métricas adicionales
        auc = roc_auc_score(all_labels, all_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, (np.array(all_probs) > 0.5).astype(int), average='binary')
        cm = confusion_matrix(all_labels, (np.array(all_probs) > 0.5).astype(int))

        metrics = {
            "val_auc": auc,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=(np.array(all_probs) > 0.5).astype(int),
                class_names=["Negative", "Positive"]
            ) if self.wandb else None
        }

        return val_loss, val_accuracy, metrics

    def _save_final_model(self, model):
        os.makedirs(self.output_model_dir, exist_ok=True)
        final_model_path = os.path.join(self.output_model_dir, 'model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")