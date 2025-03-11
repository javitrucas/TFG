import torch
import numpy as np
from scripts.model import MILModel
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, roc_curve


class ModelEvaluator:
    def __init__(self, model_path, test_loader, batch_size, wandb=None):
        self.model_path = model_path
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.wandb = wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        """
        Load the model from the saved file.
        """
        try:
            # Instantiate the model architecture
            model = MILModel(feature_dim=512).to(self.device)
            
            # Load the state dictionary
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            
            # Load the state dictionary into the model
            model.load_state_dict(state_dict)
            
            # Move the model to the appropriate device
            model.to(self.device)
            model.eval()  # Set the model to evaluation mode
            
            print(f"Model loaded successfully from {self.model_path}")
            return model
        
        except Exception as e:
            raise RuntimeError(f"Error loading model from {self.model_path}: {e}")

    def evaluate(self):
        """
        Realiza la evaluación del modelo.
        """
        model = self._load_model()  # Cargar el modelo
        model.eval()

        all_labels, all_probs = [], []

        with torch.no_grad():
            for batch in self.test_loader:
                bag_data, bag_label = batch[0], batch[1]
                bag_data, bag_label = bag_data.to(self.device), bag_label.to(self.device)

                output, _ = model(bag_data)
                probs = torch.sigmoid(output.squeeze(-1)).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(bag_label.cpu().numpy())

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calcular métricas
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        all_preds = (all_probs > optimal_threshold).astype(int)

        accuracy = np.mean(all_preds == all_labels)
        auc = roc_auc_score(all_labels, all_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
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

        self._log_metrics(metrics)
        return metrics

    def _log_metrics(self, metrics):
        """
        Registra las métricas en wandb.
        """
        print("\n--- Evaluation Results ---")
        print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])

        if self.wandb:
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