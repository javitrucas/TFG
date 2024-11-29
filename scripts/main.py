from bag_creator import BagCreator
from training import Training
from evaluation import ModelEvaluator
import os

if __name__ == "__main__":
    # Parámetros controlados dentro del código
    target_digit = 3          # Dígito objetivo para las bolsas
    num_bags = 10           # Número de bolsas
    num_instances = 10        # Número de instancias por bolsa
    num_epochs = 10           # Número de épocas de entrenamiento
    learning_rate = 1e-3      # Tasa de aprendizaje
    batch_size = 1            # Tamaño del lote (batch size)
    
    # Directorios para guardar los modelos y las gráficas
    output_bags_dir='./output/10_bags/10_inst/10_epochs/bag_creator'
    output_model_dir = './models/10_bags/10_inst/10_epochs'
    output_graphs_dir = './output/10_bags/10_inst/10_epochs/training_graphs'
    attention_dir='./output/10_bags/10_inst/10_epochs/attention_images'
    test_graphs_dir = './output/10_bags/10_inst/10_epochs/test_graphs'
    
    # Crear instancia de BagCreator para generar los datos
    bag_creator = BagCreator(target_digit=target_digit, num_bags=num_bags, num_instances=num_instances, output_bags_dir=output_bags_dir)

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
    trainer = Training(
        train_bags, train_labels, val_bags, val_labels, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate, 
        output_model_dir=output_model_dir, 
        output_graphs_dir=output_graphs_dir
    )
    trainer.train()

    # Iniciar la evaluación
    evaluator = ModelEvaluator(
        model_path=os.path.join(output_model_dir, 'model.pth'),
        test_bags=eval_bags, 
        test_labels=eval_labels, 
        output_graphs_dir=test_graphs_dir,
        attention_dir=attention_dir,  
        batch_size=batch_size
    )
    evaluator.evaluate()