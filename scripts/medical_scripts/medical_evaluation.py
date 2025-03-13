import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, roc_curve
from scripts.model import MILModel  # Asegúrate de que el modelo incluya pooling_type

class ModelEvaluator:
    def __init__(
        self, 
        model_path, 
        test_loader, 
        batch_size, 
        pooling_type='attention',  # Añadir pooling_type
        wandb=None
    ):
        self.model_path = model_path
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.pooling_type = pooling_type  # Almacenar el tipo de pooling
        self.wandb = wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inicializar listas para métricas
        self.test_loss_curve = []
        self.test_accuracy_curve = []
        self.test_recall = []
        self.test_precision = []
        self.test_auc_roc = []
        self.test_f1_score = []

    def _plot_attention_heatmap(self, attention_weights):
        """
        Genera un gráfico de barras para visualizar pesos de atención.
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Crear gráfico de barras
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = np.arange(len(attention_weights))
        ax.bar(indices, attention_weights, color='blue', alpha=0.7)
        ax.set_title("Attention Weights per Instance")
        ax.set_xlabel("Instance Index")
        ax.set_ylabel("Attention Weight")
        ax.set_xticks(indices)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        return fig

    def _load_model(self):
        """
        Carga el modelo con el pooling_type correcto.
        """
        try:
            # Inicializar el modelo con el pooling_type
            model = MILModel(
                feature_dim=512,  # Ajustar según el paper (ver Tabla 8/9/15/16)
                pooling_type=self.pooling_type
            ).to(self.device)
            
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def evaluate(self):
        """
        Realiza la evaluación y devuelve métricas + pesos de atención (si aplica).
        """
        model = self._load_model()
        criterion = torch.nn.BCEWithLogitsLoss()
        all_labels, all_probs, attention_weights_list = [], [], []

        with torch.no_grad():
            for batch in self.test_loader:
                # Desempaquetar batch según el formato del dataset
                bag_data, bag_label, _, adj_mat, mask = batch  # Ajustar según el dataset_loader
                bag_data = bag_data.to(self.device)
                bag_label = bag_label.to(self.device)
                mask = mask.to(self.device) if mask is not None else None
                adj_mat = adj_mat.to(self.device) if adj_mat is not None else None

                # Forward pass (incluir mask y adj_mat)
                output, attention_weights = model(bag_data, mask=mask, adj_mat=adj_mat)
                
                # Calcular loss
                loss = criterion(output.squeeze(-1), bag_label.float())
                self.test_loss_curve.append(loss.item())

                # Obtener probabilidades y labels
                probs = torch.sigmoid(output.squeeze(-1)).cpu().numpy()
                labels = bag_label.cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels)

                # Almacenar pesos de atención solo si es necesario
                if self.pooling_type == 'attention' and attention_weights is not None:
                    attention_weights_list.append(attention_weights.cpu().numpy())

        # Calcular métricas finales
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calcular optimal threshold usando ROC
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        all_preds = (all_probs > optimal_threshold).astype(int)

        accuracy = np.mean(all_preds == all_labels)
        auc = roc_auc_score(all_labels, all_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, 
            all_preds, 
            average='binary'
        )
        cm = confusion_matrix(all_labels, all_preds)

        metrics = {
            "accuracy": accuracy,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "all_labels": all_labels,
            "all_preds": all_preds,
            "optimal_threshold": optimal_threshold
        }

        # Registrar métricas en wandb
        self._log_metrics(metrics)
        
        #self._plot_attention_heatmap(attention_weights)

        # Devolver métricas y pesos de atención (solo si aplica)
        return metrics, attention_weights_list

    def _log_metrics(self, metrics):
        """
        Registra métricas en wandb y muestra resultados.
        """
        print("\n--- Evaluation Results ---")
        print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])

        if self.wandb:
            # Registrar métricas escalares
            self.wandb.log({
                "test_accuracy": metrics["accuracy"],
                "test_auc": metrics["auc"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1_score"],
                "confusion_matrix": self.wandb.plot.confusion_matrix(
                    preds=metrics["all_preds"],
                    y_true=metrics["all_labels"],
                    class_names=["Negative", "Positive"]
                )
            })

            # Registrar heatmaps de atención si existen
            if self.pooling_type == 'attention':
                for i, weights in enumerate(metrics.get("attention_weights", [])):
                    self.wandb.log({
                        f"attention_weights_bag_{i}": self.wandb.Image(
                            self._plot_attention_heatmap(weights)
                        )
                    })