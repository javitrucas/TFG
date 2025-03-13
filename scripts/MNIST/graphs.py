import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

class Graphs:
    def __init__(self, num_epochs, train_losses, val_losses, train_accuracies, val_accuracies, output_dir='./output/graphs'):
        """
        Inicializa la clase Graphs.

        :param num_epochs: Número de épocas de entrenamiento.
        :param train_losses: Lista con las pérdidas de entrenamiento por época.
        :param val_losses: Lista con las pérdidas de validación por época.
        :param train_accuracies: Lista con las precisiones de entrenamiento por época.
        :param val_accuracies: Lista con las precisiones de validación por época.
        :param output_dir: Directorio donde se guardarán las gráficas.
        """
        self.num_epochs = num_epochs
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_accuracies = train_accuracies
        self.val_accuracies = val_accuracies
        self.output_dir = output_dir
        
        # Crear directorio de salida si no existe
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_loss(self):
        """
        Graficar la pérdida durante el entrenamiento y la validación.
        """
        plt.plot(range(1, self.num_epochs + 1), self.train_losses, marker='o', linestyle='-', color='b', label='Train Loss')
        plt.plot(range(1, self.num_epochs + 1), self.val_losses, marker='x', linestyle='--', color='r', label='Validation Loss')
        plt.title('Pérdida durante el entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'training_loss.png'))
        plt.close()

    def plot_accuracy(self):
        """
        Graficar la precisión durante el entrenamiento y la validación.
        """
        plt.plot(range(1, self.num_epochs + 1), self.train_accuracies, marker='o', linestyle='-', color='g', label='Train Accuracy')
        plt.plot(range(1, self.num_epochs + 1), self.val_accuracies, marker='x', linestyle='--', color='orange', label='Validation Accuracy')
        plt.title('Precisión durante el entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'training_accuracy.png'))
        plt.close()

    def plot_roc_curve(self, y_true, y_scores):
        """
        Graficar la curva ROC.
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'))
        plt.close()

    def plot_precision_recall_curve(self, y_true, y_scores):
        """
        Graficar la curva de precisión-recall.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'precision_recall_curve.png'))
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Graficar la matriz de confusión.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()

    def show_plots(self):
        """
        Mostrar las gráficas en pantalla.
        """
        plt.show()

    def save_plots(self):
        """
        Guardar las gráficas de entrenamiento y validación.
        """
        self.plot_loss()
        self.plot_accuracy()