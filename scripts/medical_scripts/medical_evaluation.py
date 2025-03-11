import torch
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

from scripts.model import MILModel


class ModelEvaluator:
    def __init__(self, model_path, test_loader, batch_size, wandb=None):
        self.model_path = model_path
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.wandb = wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):
        # Cargar el modelo
        model = MILModel(feature_dim=512).to(self.device)
        model.load_state_dict(torch.load(self.model_path, weights_only=True))  # Usar weights_only=True por seguridad
        model.eval()

        all_labels, all_probs, all_preds = [], [], []

        with torch.no_grad():
            for bag_data, bag_label, _, _, _ in self.test_loader:
                bag_data, bag_label = bag_data.to(self.device), bag_label.to(self.device)

                # Obtener la salida del modelo
                output, _ = model(bag_data)  # Desempaquetar la tupla (output, attention_weights)
                probs = torch.sigmoid(output.squeeze(-1))  # Calcular probabilidades y eliminar dimensión innecesaria
                preds = (probs > 0.5).float()  # Clasificación binaria

                # Guardar etiquetas, probabilidades y predicciones
                all_labels.extend(bag_label.cpu().numpy())
                all_probs.extend(probs.cpu().numpy().flatten())  # Asegurarse de que sea iterable
                all_preds.extend(preds.cpu().numpy().flatten())

        # Convertir a arrays de NumPy para facilitar el cálculo de métricas
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)

        # Calcular métricas
        accuracy = np.mean(all_preds == all_labels)  # Precisión como promedio de aciertos
        auc = roc_auc_score(all_labels, all_probs)  # Área bajo la curva ROC
        cm = confusion_matrix(all_labels, all_preds)  # Matriz de confusión

        # Registrar métricas en wandb
        if self.wandb:
            self.wandb.log({
                "test_accuracy": accuracy,
                "test_auc": auc,
                "confusion_matrix": wandb.plot.confusion_matrix(
                    preds=all_preds.astype(int),  # Convertir a enteros
                    y_true=all_labels.astype(int),
                    class_names=["Negative", "Positive"]
                )
            })

        print(f"Test Accuracy: {accuracy:.4f}, Test AUC: {auc:.4f}")