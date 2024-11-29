import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, f1_score
from model import MILModel
from bag_creator import BagCreator
from graphs import Graphs
from training import Training

class ModelEvaluator:
    def __init__(self, model_path, test_bags, test_labels, output_graphs_dir='./output/10_inst/test_graphs', attention_dir='./output/10_inst/attention_images', batch_size=1):
        self.model = MILModel()  # Asegúrate de tener la clase MILModel importada correctamente
        self.model.load_state_dict(torch.load(model_path))  # Cargar el modelo entrenado
        self.model.eval()  # Establecer el modelo en modo evaluación
        self.test_bags = test_bags
        self.test_labels = test_labels

        # Crear el conjunto de datos de evaluación
        self.test_dataset = TensorDataset(torch.stack(self.test_bags), self.test_labels)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        # Función de pérdida
        self.criterion = nn.BCELoss()

        # Directorio de salida para las gráficas
        self.output_graphs_dir = output_graphs_dir
        self.attention_dir = attention_dir
        os.makedirs(self.output_graphs_dir, exist_ok=True)
        os.makedirs(self.attention_dir, exist_ok=True)

    def save_attention_images(self, bag, attention_scores, label, bag_id):
        fig, axes = plt.subplots(1, len(bag), figsize=(15, 5))
        fig.suptitle(f'Bag {bag_id} - Prediction: {label}', fontsize=16)

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
            for bag_id, (bag, label) in enumerate(self.test_dataloader):
                output, attention_scores = self.model(bag[0])  # bag[0] porque viene empaquetado con DataLoader
                loss = self.criterion(output, label.float().unsqueeze(1))
                
                test_loss += loss.item()
                
                # Calcular la precisión de evaluación
                predicted = (output > 0.5).float()  # Umbral de 0.5 para clasificación binaria
                correct_test += (predicted == label.unsqueeze(1)).sum().item()
                total_test += label.size(0)
                
                predictions.append(output.item())  # Guardar las predicciones
                true_labels.append(label.item())  # Guardar las etiquetas verdaderas

                # Guardar imágenes con atenciones
                if count_0 < 3 and label==0:
                    self.save_attention_images(bag[0], attention_scores, output.item(), bag_id)
                    count_0=count_0+1
                elif count_1 < 3 and label==1:
                    self.save_attention_images(bag[0], attention_scores, output.item(), bag_id)
                    count_1=count_1+1


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
    # Crear instancia de BagCreator para generar los datos
    bag_creator = BagCreator(target_digit=3, num_bags=1000, num_instances=10)

    # Crear las bolsas y obtener las etiquetas
    bags, labels = bag_creator.create_bags()

    # Dividir las bolsas en entrenamiento (70%) y evaluación (30%)
    split_idx = int(len(bags) * 0.7)  # 70% para entrenamiento y 30% para evaluación
    train_bags, eval_bags = bags[:split_idx], bags[split_idx:]
    train_labels, eval_labels = labels[:split_idx], labels[split_idx:]

    # Dividir el conjunto de entrenamiento (70%) en entrenamiento (80%) y validación (20%)
    train_split_idx = int(len(train_bags) * 0.8)  # 80% de entrenamiento y 20% de validación
    train_bags, val_bags = train_bags[:train_split_idx], train_bags[train_split_idx:]
    train_labels, val_labels = train_labels[:train_split_idx], train_labels[train_split_idx:]

    # Iniciar el entrenamiento
    trainer = Training(train_bags, train_labels, val_bags, val_labels, num_epochs=10, learning_rate=1e-3, output_model_dir='./models', output_graphs_dir = './output/10_inst/training_graphs')
    trainer.train()

    # Iniciar la evaluación
    evaluator = ModelEvaluator(model_path='./models/10_inst/model.pth', test_bags=eval_bags, test_labels=eval_labels, output_graphs_dir = './output/10_inst/test_graphs', attention_dir='./output/10_inst/attention_images', batch_size=1)
    evaluator.evaluate()