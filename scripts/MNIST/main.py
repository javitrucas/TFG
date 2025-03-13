# Importar MNISTMILDataset
import os
import torch
from MNISTMILDataset import MNISTMILDataset
from evaluation import ModelEvaluator
from training import Training

if __name__ == "__main__":
    # Parámetros controlados dentro del código
    target_digit = 3          # Dígito objetivo para las bolsas
    bag_size = 10             # Número de instancias por bolsa
    num_epochs = 10           # Número de épocas de entrenamiento
    learning_rate = 1e-3      # Tasa de aprendizaje
    batch_size = 1            # Tamaño del lote (batch size)
    pooling_type = 'mean'  # Tipo de agrupación (attention, mean, max)
    
    # Directorios para guardar los modelos y las gráficas
    output_model_dir = './models/MNIST/new'
    output_graphs_dir = './output/MNIST/new/training_graphs'
    attention_dir = './output/MNIST/new/attention_images'
    test_graphs_dir = './output/MNIST/new/test_graphs'
    
    # Crear instancia de MNISTMILDataset para generar los datos
    train_dataset = MNISTMILDataset(subset="train", bag_size=bag_size, obj_label=target_digit)
    test_dataset = MNISTMILDataset(subset="test", bag_size=bag_size, obj_label=target_digit)
    
    # Dividir el conjunto de entrenamiento en entrenamiento (80%) y validación (20%)
    train_split_idx = int(len(train_dataset) * 0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_split_idx, len(train_dataset) - train_split_idx])
    
    # Iniciar el entrenamiento
    trainer = Training(
        train_dataset=train_dataset, 
        #train_labels=None,  # Ya no necesitamos pasar etiquetas separadas
        val_dataset=val_dataset, 
        #val_labels=None, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate, 
        output_model_dir=output_model_dir, 
        output_graphs_dir=output_graphs_dir,
        pooling_type=pooling_type
    )
    trainer.train()
    
    # Iniciar la evaluación
    evaluator = ModelEvaluator(
        model_path=os.path.join(output_model_dir, 'model.pth'),
        test_dataset=test_dataset, 
        #test_labels=None,  # Ya no necesitamos pasar etiquetas separadas
        output_graphs_dir=test_graphs_dir,
        attention_dir=attention_dir,  
        batch_size=batch_size,
        pooling_type=pooling_type
    )
    evaluator.evaluate()