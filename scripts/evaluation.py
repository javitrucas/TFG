import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, f1_score
from model import MILModel
from MNISTMILDataset import MNISTMILDataset
from MIL_utils import MIL_collate_fn
from graphs import Graphs

class ModelEvaluator:
    def __init__(self, model_path, test_dataset, output_graphs_dir='./output/test_graphs', attention_dir='./output/attention_images', batch_size=1):
        # Cargar el modelo entrenado
        self.model = MILModel()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()  # Establecer el modelo en modo evaluación
        
        # Usar el dataset directamente
        self.test_dataset = test_dataset
        
        # Crear DataLoader con MIL_collate_fn
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=MIL_collate_fn
        )
        
        # Función de pérdida
        self.criterion = nn.BCELoss()
        
        # Directorios de salida
        self.output_graphs_dir = output_graphs_dir
        self.attention_dir = attention_dir
        os.makedirs(self.output_graphs_dir, exist_ok=True)
        os.makedirs(self.attention_dir, exist_ok=True)

    def save_attention_images(self, bag, attention_scores, label, bag_id):
        fig, axes = plt.subplots(1, len(bag), figsize=(15, 5))
        fig.suptitle(f'Bag {bag_id} - Prediction: {label:.2f}', fontsize=16)

        for i, (img, attn) in enumerate(zip(bag, attention_scores)):
            img = img.squeeze(0).numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Attn: {attn.item():.2f}')

        output_path = os.path.join(self.attention_dir, f'bag_{bag_id}.png')
        plt.savefig(output_path)
        plt.close(fig)

    def evaluate(self):
        test_loss = 0
        correct_test = 0
        total_test = 0
        predictions = []
        true_labels = []
        count_0 = 0
        count_1 = 0

        # Evaluación del modelo
        with torch.no_grad():  # No se necesitan gradientes durante la evaluación
            for bag_id, (bag_data, bag_label, inst_labels, adj_mat, mask) in enumerate(self.test_dataloader):
                # Forward pass
                output, attention_scores = self.model(bag_data, mask, adj_mat)
                loss = self.criterion(output, bag_label.unsqueeze(1))
                
                test_loss += loss.item()
                
                # Calcular la precisión de evaluación
                predicted = (output > 0.5).float()  # Umbral de 0.5 para clasificación binaria
                correct_test += (predicted == bag_label.unsqueeze(1)).sum().item()
                total_test += bag_label.size(0)
                
                predictions.append(output.item())  # Guardar las predicciones
                true_labels.append(bag_label.item())  # Guardar las etiquetas verdaderas

                # Guardar imágenes con atenciones
                if count_0 < 3 and bag_label == 0:
                    self.save_attention_images(bag_data[0], attention_scores[0], output.item(), bag_id)
                    count_0 += 1
                elif count_1 < 3 and bag_label == 1:
                    self.save_attention_images(bag_data[0], attention_scores[0], output.item(), bag_id)
                    count_1 += 1

        # Calcular el F1-score
        f1 = f1_score(true_labels, [1 if p > 0.5 else 0 for p in predictions])
        print(f"F1-Score: {f1:.4f}")

        # Crear instancia de la clase Graphs
        graphs = Graphs(
            num_epochs=1,  # No se usan épocas en la evaluación, pero se puede pasar cualquier valor
            train_losses=[],  # Las pérdidas de entrenamiento no se utilizan aquí
            val_losses=[],  # Las pérdidas de validación no se utilizan aquí
            train_accuracies=[],  # Las precisiones de entrenamiento no se utilizan aquí
            val_accuracies=[],  # Las precisiones de validación no se utilizan aquí
            output_dir=self.output_graphs_dir
        )

        # Graficar la curva ROC
        graphs.plot_roc_curve(true_labels, predictions)

        # Graficar la curva de precisión-recall
        graphs.plot_precision_recall_curve(true_labels, predictions)

        # Graficar la matriz de confusión
        graphs.plot_confusion_matrix(true_labels, [1 if p > 0.5 else 0 for p in predictions])

        # Mostrar las gráficas
        graphs.show_plots()

if __name__ == "__main__":
    # Crear instancia de MNISTMILDataset para generar los datos
    test_dataset = MNISTMILDataset(subset="test", bag_size=10, obj_label=3)

    # Iniciar la evaluación
    evaluator = ModelEvaluator(
        model_path='./models/model.pth',
        test_dataset=test_dataset,
        output_graphs_dir='./output/test_graphs',
        attention_dir='./output/attention_images',
        batch_size=1
    )
    evaluator.evaluate()