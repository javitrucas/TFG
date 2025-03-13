import os
import torch
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
from scripts.model import MILModel  # Asegúrate de que MILModel acepte pooling_type

class Training:
    def __init__(
        self, 
        train_loader, 
        val_loader, 
        num_epochs, 
        learning_rate, 
        output_model_dir, 
        pooling_type='attention',  # Añadir pooling_type
        patience=5, 
        wandb=None
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_model_dir = output_model_dir
        self.pooling_type = pooling_type  # Almacenar el tipo de pooling
        self.patience = patience
        self.wandb = wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Inicializar el modelo con el pooling_type
        self.model = MILModel(pooling_type=self.pooling_type, feature_dim=512).to(self.device)
        
        # Listas para métricas
        self.train_loss_curve = []
        self.val_loss_curve = []
        self.train_accuracy_curve = []
        self.val_accuracy_curve = []
        self.train_auc_roc = []
        self.val_auc_roc = []
        self.train_f1_score = []
        self.val_f1_score = []
        self.train_precision = []
        self.train_recall = []
        self.val_precision = []
        self.val_recall = []

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        best_val_auc = 0.0
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            # Entrenamiento
            train_metrics = self._train_epoch(optimizer, criterion)
            self.train_loss_curve.append(train_metrics['loss'])
            self.train_accuracy_curve.append(train_metrics['accuracy'])
            self.train_auc_roc.append(train_metrics['auc'])
            self.train_f1_score.append(train_metrics['f1'])
            
            # Validación
            val_metrics = self._validate_epoch(criterion)
            self.val_loss_curve.append(val_metrics['loss'])
            self.val_accuracy_curve.append(val_metrics['accuracy'])
            self.val_auc_roc.append(val_metrics['auc'])
            self.val_f1_score.append(val_metrics['f1'])
            self.val_precision.append(val_metrics['precision'])
            self.val_recall.append(val_metrics['recall'])

            # Registro en wandb
            if self.wandb:
                self.wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_metrics['loss'],
                    "train_accuracy": train_metrics['accuracy'],
                    "train_auc": train_metrics['auc'],
                    "train_f1": train_metrics['f1'],
                    "val_loss": val_metrics['loss'],
                    "val_accuracy": val_metrics['accuracy'],
                    "val_auc": val_metrics['auc'],
                    "val_f1": val_metrics['f1'],
                    "val_confusion_matrix": val_metrics['confusion_matrix'],
                    "pooling_type": self.pooling_type  # Registrar el tipo de pooling
                })

            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")

            # Early stopping basado en AUC de validación
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                epochs_no_improve = 0
                self._save_final_model()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print("Early stopping triggered.")
                    break

    def _train_epoch(self, optimizer, criterion):
        self.model.train()
        train_loss = 0.0
        all_labels, all_probs = [], []

        for bag_data, bag_label, _, _, mask in self.train_loader:
            bag_data = bag_data.to(self.device)
            bag_label = bag_label.to(self.device)
            mask = mask.to(self.device) if mask is not None else None

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                output, _ = self.model(bag_data, mask=mask)  # Ignorar attention_weights en entrenamiento
                loss = criterion(output, bag_label.float().unsqueeze(1))
            
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            probs = torch.sigmoid(output).squeeze(-1).detach().cpu().numpy()
            labels = bag_label.cpu().numpy()
            
            train_loss += loss.item()
            all_labels.extend(labels)
            all_probs.extend(probs)

        # Calcular métricas de entrenamiento
        auc = roc_auc_score(all_labels, all_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, 
            (np.array(all_probs) > 0.5).astype(int), 
            average='binary'
        )
        accuracy = np.mean((np.array(all_probs) > 0.5).astype(int) == all_labels)

        return {
            'loss': train_loss / len(self.train_loader),
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1
        }

    def _validate_epoch(self, criterion):
        self.model.eval()
        val_loss = 0.0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for bag_data, bag_label, _, _, mask in self.val_loader:
                bag_data = bag_data.to(self.device)
                bag_label = bag_label.to(self.device)
                mask = mask.to(self.device) if mask is not None else None

                output, _ = self.model(bag_data, mask=mask)  # Ignorar attention_weights en validación
                loss = criterion(output, bag_label.float().unsqueeze(1))
                
                probs = torch.sigmoid(output).squeeze(-1).cpu().numpy()
                labels = bag_label.cpu().numpy()
                
                val_loss += loss.item()
                all_labels.extend(labels)
                all_probs.extend(probs)

        # Calcular métricas de validación
        auc = roc_auc_score(all_labels, all_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, 
            (np.array(all_probs) > 0.5).astype(int), 
            average='binary'
        )
        accuracy = np.mean((np.array(all_probs) > 0.5).astype(int) == all_labels)
        cm = confusion_matrix(all_labels, (np.array(all_probs) > 0.5).astype(int))

        return {
            'loss': val_loss / len(self.val_loader),
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': self.wandb.plot.confusion_matrix(
                y_true=all_labels,
                preds=(np.array(all_probs) > 0.5).astype(int),
                class_names=["Negative", "Positive"]
            ) if self.wandb else None
        }

    def _save_final_model(self):
        os.makedirs(self.output_model_dir, exist_ok=True)
        model_path = os.path.join(self.output_model_dir, f'model_{self.pooling_type}.pth')  # Nombre único por pooling_type
        torch.save(self.model.state_dict(), model_path)
        print(f"Final model saved to {model_path}")