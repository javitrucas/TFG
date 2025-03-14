import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, roc_curve
from scripts.model import MILModel  # Asegúrate de que el modelo incluya pooling_type
from scripts.medical_scripts.visualization_helper import VisualizationHelper  # Importar la nueva clase

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

        # Crear directorio de salida según el tipo de pooling
        self.output_dir = f"output/{self.pooling_type}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Inicializar el helper de visualización
        self.visualization_helper = VisualizationHelper(self.output_dir)

    def _plot_attention_heatmap(self, attention_weights):
        """
        Genera un heatmap de atención para visualización y guardado.
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.imshow(attention_weights, cmap='viridis', aspect='auto')
        fig.colorbar(cax, ax=ax)
        ax.set_title("Attention Weights")
        ax.set_xlabel("Instances")
        ax.set_ylabel("Bags")
        return fig

    def _save_confusion_matrix(self, cm, filename):
        """
        Guarda la matriz de confusión como una imagen.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.matshow(cm, cmap='Blues')
        fig.colorbar(cax)

        # Etiquetas
        ax.set_xticklabels([''] + ["Negative", "Positive"])
        ax.set_yticklabels([''] + ["Negative", "Positive"])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        # Guardar la matriz de confusión
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close(fig)
        print(f"Confusion matrix saved at {filepath}")

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
        all_labels, all_probs, attention_weights_list, all_patches = [], [], [], []

        with torch.no_grad():
            for batch in self.test_loader:
                # Desempaquetar batch según el formato del dataset
                bag_data, bag_label, patches, adj_mat, mask = batch  # Asumimos que 'patches' está disponible
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

                # Almacenar pesos de atención y parches solo si es necesario
                if self.pooling_type == 'attention' and attention_weights is not None:
                    attention_weights_list.append(attention_weights.cpu().numpy())
                    all_patches.append(patches)  # Guardar los parches para visualización

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

        # Guardar la matriz de confusión
        self._save_confusion_matrix(cm, "confusion_matrix.png")

        # Registrar métricas en wandb
        self._log_metrics(metrics)
        
        # Guardar heatmap de atención si aplica
        if self.pooling_type == 'attention':
            for i, (weights, patches) in enumerate(zip(attention_weights_list[:5], all_patches[:5])):
                fig = self._plot_attention_heatmap(weights)
                filepath = os.path.join(self.output_dir, f"attention_heatmap_{i}.png")
                fig.savefig(filepath)
                plt.close(fig)
                print(f"Attention heatmap {i} saved at {filepath}")

                # Visualizar atención en las imágenes originales
                # image = self._reconstruct_image_from_patches(patches)  # Reconstruir la imagen original
                # self.visualization_helper.plot_attention_on_image(image, patches, weights, bag_id=i)

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

    def _reconstruct_image_from_patches(self, patches):
        """
        Reconstruye la imagen original a partir de los parches.
        Esto depende de cómo estén organizados tus datos.
        """
        # Implementar la lógica para reconstruir la imagen aquí
        raise NotImplementedError("Implementa la reconstrucción de la imagen según tu estructura de datos.")