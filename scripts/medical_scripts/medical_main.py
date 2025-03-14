import os
import sys
import torch
import wandb
import torch.optim as optim
from pathlib import Path

# Agregar el directorio raíz al PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.dataset_loader import load_dataset
from medical_evaluation import ModelEvaluator
from medical_training import Training
from scripts.MIL_utils import MIL_collate_fn

# Configuración de hiperparámetros y pooling_type
CONFIGS = [
    {
        "dataset_name": "rsna-features_resnet18",
        "num_epochs": 3,
        "learning_rate": 1e-3,
        "batch_size": 1,
        "val_prop": 0.15,
        "seed": 42,
        "use_inst_distances": False,
        "adj_mat_mode": "relative",
        "pooling_type": "attention"  # Tipo de pooling: attention, mean, max
    },
    {
        "dataset_name": "rsna-features_resnet18",
        "num_epochs": 5,
        "learning_rate": 1e-3,
        "batch_size": 1,
        "val_prop": 0.15,
        "seed": 42,
        "use_inst_distances": False,
        "adj_mat_mode": "relative",
        "pooling_type": "mean"
    },
    {
        "dataset_name": "rsna-features_resnet18",
        "num_epochs": 5,
        "learning_rate": 1e-3,
        "batch_size": 1,
        "val_prop": 0.15,
        "seed": 42,
        "use_inst_distances": False,
        "adj_mat_mode": "relative",
        "pooling_type": "max"
    }
]

for config_dict in CONFIGS:
    # Inicializar wandb con configuración única por pooling_type
    wandb.init(
        project="TFG",
        config=config_dict,
        reinit=True  # Reiniciar wandb para múltiples ejecuciones
    )
    config = wandb.config

    print(f"=== Iniciando experimento con pooling_type={config.pooling_type} ===")

    # Directorio para guardar modelos (único por pooling_type)
    output_model_dir = f"./models/{config.dataset_name.split('-')[0]}_{config.pooling_type}"
    os.makedirs(output_model_dir, exist_ok=True)
    model_path = os.path.join(output_model_dir, f"model_{config.pooling_type}.pth")

    # Cargar datasets
    train_dataset, val_dataset = load_dataset(config=config, mode="train_val")
    test_dataset = load_dataset(config=config, mode="test")

    # Crear dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=MIL_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=MIL_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=MIL_collate_fn)

    # Entrenamiento
    trainer = Training(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        output_model_dir=output_model_dir,
        pooling_type=config.pooling_type,  # Pasar el tipo de pooling
        wandb=wandb
    )
    trainer.train()

    # Guardar modelo con nombre único
    torch.save(trainer.model.state_dict(), model_path)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    print(model_path)

    # Evaluación
    evaluator = ModelEvaluator(
        model_path=model_path,
        test_loader=test_loader,
        batch_size=config.batch_size,
        pooling_type=config.pooling_type,  # Pasar el tipo de pooling
        wandb=wandb
    )
    evaluator.evaluate()

    # Finalizar wandb para este experimento
    wandb.finish()