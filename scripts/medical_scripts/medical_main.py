import os
import sys
import torch
import wandb  # Importar wandb
import torch.optim as optim

# Agregar el directorio raíz al PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

from scripts.dataset_loader import load_dataset
from medical_evaluation import ModelEvaluator
from medical_training import Training
from scripts.MIL_utils import MIL_collate_fn

if __name__ == "__main__":
    # Iniciar wandb
    wandb.init(
        project="TFG",  # Nombre del proyecto en wandb
        config={
            "dataset_name": "rsna-features_resnet18",  # Nombre del dataset
            "num_epochs": 25,                         # Número de épocas
            "learning_rate": 1e-3,                   # Tasa de aprendizaje
            "batch_size": 1,                         # Tamaño del lote
            "val_prop": 0.15,                         # Proporción de validación
            "seed": 42,                               # Semilla para reproducibilidad
            "use_inst_distances": False,  
            "adj_mat_mode": "relative"          # No lo uso, pero tengo que poner algo
        }
    )

    # Parámetros controlados dentro del código
    config = wandb.config  # Usar la configuración de wandb
    dataset_name = config.dataset_name
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    batch_size = config.batch_size

    # Directorio para guardar modelos
    output_model_dir = f"./models/{dataset_name.split('-')[0]}"
    os.makedirs(output_model_dir, exist_ok=True)

    # Cargar datasets usando dataset_loader.py
    train_dataset, val_dataset = load_dataset(config=config, mode="train_val")
    test_dataset = load_dataset(config=config, mode="test")

    # Crear dataloaders con MIL_collate_fn
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=MIL_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=MIL_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=MIL_collate_fn)

    # Iniciar el entrenamiento
    trainer = Training(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        output_model_dir=output_model_dir,
        wandb=wandb  # Pasar wandb al trainer
    )
    trainer.train()

    # Guardar el modelo como un artefacto en wandb
    model_path = os.path.join(output_model_dir, 'model.pth')
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    # Iniciar la evaluación
    evaluator = ModelEvaluator(
        model_path=model_path,
        test_loader=test_loader,
        batch_size=batch_size,
        wandb=wandb  # Pasar wandb al evaluator
    )
    evaluator.evaluate()

    # Finalizar wandb
    wandb.finish()