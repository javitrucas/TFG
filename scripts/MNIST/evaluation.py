import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import sys
print(sys.path)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, f1_score, roc_auc_score

from scripts.MNIST.MNISTmodel import MILModel
from scripts.MNIST.MNISTMILDataset import MNISTMILDataset
from scripts.MIL_utils import MIL_collate_fn
from scripts.MNIST.graphs import Graphs

class ModelEvaluator:
    def __init__(self, model_path, test_dataset, output_graphs_dir='./output/test_graphs', attention_dir='./output/attention_images', batch_size=1, pooling_type='attention'):
        # Cargar el modelo entrenado
        self.model = MILModel(pooling_type)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()  # Establecer el modelo en modo evaluación
        
        if(pooling_type == 'attention'):
            output_graphs_dir = output_graphs_dir + '/attention'
            attention_dir = attention_dir + '/attention'
        elif(pooling_type == 'mean'):
            output_graphs_dir = output_graphs_dir + '/mean'
            attention_dir = attention_dir + '/mean'
        elif(pooling_type == 'max'):
            output_graphs_dir = output_graphs_dir + '/max'
            attention_dir = attention_dir + '/max'

        
        
        # Usar el dataset directamente
        self.test_dataset = test_dataset
        self.pooling_type = pooling_type

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
        saved_counts = {0: 0, 1: 0}
        test_loss = 0
        correct_test = 0
        total_test = 0
        predictions = []
        true_labels = []
        attention_weights = []  # Almacenar pesos de atención si existen

        with torch.no_grad():
            for bag_id, (bag_data, bag_label, inst_labels, adj_mat, mask) in enumerate(self.test_dataloader):
                output, attention_scores = self.model(bag_data, mask, adj_mat)
                loss = self.criterion(output, bag_label.unsqueeze(1))
                
                test_loss += loss.item()
                predicted = (output > 0.5).float()
                correct_test += (predicted == bag_label.unsqueeze(1)).sum().item()
                total_test += bag_label.size(0)
                
                predictions.append(output.item())
                true_labels.append(bag_label.item())

                # Guardar imágenes de atención (solo si hay pesos)
                if attention_scores is not None:
                    # Extraer la etiqueta como entero
                    label = bag_label.item()
                    # Si la etiqueta es 0 o 1 y aun no hemos guardado 3 imágenes para esa clase, se guarda
                    if label in saved_counts and saved_counts[label] < 3:
                        self.save_attention_images(bag_data[0], attention_scores[0], output.item(), bag_id)
                        saved_counts[label] += 1
                    # Almacenar pesos de atención para devolverlos
                    attention_weights.append(attention_scores.detach().cpu().numpy())
        
        # Cálculo de métricas
        accuracy = correct_test / total_test
        f1 = f1_score(true_labels, [1 if p > 0.5 else 0 for p in predictions])
        auc_score = roc_auc_score(true_labels, predictions)
        
        # Crear y mostrar gráficas
        graphs = Graphs(
            num_epochs=1,
            train_losses=[],
            val_losses=[],
            train_accuracies=[],
            val_accuracies=[],
            output_dir=self.output_graphs_dir
        )
        graphs.plot_roc_curve(true_labels, predictions)
        graphs.plot_precision_recall_curve(true_labels, predictions)
        graphs.plot_confusion_matrix(true_labels, [1 if p > 0.5 else 0 for p in predictions])
        graphs.show_plots()

        results = {
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc_score,
            "test_loss": test_loss / len(self.test_dataloader)
        }
        
        return results, attention_weights
    
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
    