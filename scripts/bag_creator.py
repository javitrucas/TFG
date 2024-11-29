import torch
import random
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

class BagCreator:
    def __init__(self, target_digit=3, num_bags=1000, num_instances=10, transform=None, output_bags_dir='./output/10_inst/bag_creator'):
        # Configuración de la semilla para reproducibilidad
        random.seed(42)
        torch.manual_seed(42)

        # Inicializar parámetros
        self.target_digit = target_digit
        self.num_bags = num_bags
        self.num_instances = num_instances
        
        # Transformación para convertir las imágenes a tensores y normalizarlas
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Media y desviación estándar
        ])
        
        # Cargar el conjunto de datos MNIST
        self.mnist = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)

        self.output_bags_dir=output_bags_dir
        os.makedirs(self.output_bags_dir, exist_ok=True)

    def create_bag(self):
        indices = random.sample(range(len(self.mnist)), self.num_instances)
        bag = [self.mnist[i][0] for i in indices]  # Obtener las imágenes
        bag_labels = [self.mnist[i][1] for i in indices]  # Obtener las etiquetas de las instancias en la bolsa
        
        # Verificar si alguna imagen de la bolsa tiene el dígito objetivo (3)
        bag_label = 1 if any(label == self.target_digit for label in bag_labels) else 0  # Si contiene un '3', asignamos etiqueta 1, de lo contrario 0
        
        return torch.stack(bag), bag_labels, bag_label

    def create_bags(self):
        bags = []
        labels = []
        count_1 = 0  # Contador de bolsas con etiqueta 1
        count_0 = 0  # Contador de bolsas con etiqueta 0
        
        for bag_id in range(self.num_bags):
            bag, bag_labels, label = self.create_bag()
            bags.append(bag)
            labels.append(label)
            
            if label==0 and count_0 < 3:
                self.save_bag_images(bag, bag_labels, label, bag_id)
            elif label == 1 and count_1 < 3:
                self.save_bag_images(bag, bag_labels, label, bag_id)
            
            # Actualizar los contadores de etiquetas
            if label == 1:
                count_1 += 1
            else:
                count_0 += 1
            
            print(f"Instancias en la bolsa: {{{', '.join(map(str, bag_labels))}}}")
            print(f"Etiqueta asignada a la bolsa: {label}\n")
        
        # Verificación de la longitud de bags y labels
        print(f"Total de bolsas: {len(bags)}")
        print(f"Total de etiquetas: {len(labels)}")

        # Verificar que ambas listas tengan la misma longitud
        assert len(bags) == len(labels), "Las listas de bolsas y etiquetas no tienen el mismo tamaño."

        # Imprimir el conteo de bolsas con etiqueta 1 y 0
        print(f"Total de bolsas con etiqueta 1: {count_1}")
        print(f"Total de bolsas con etiqueta 0: {count_0}")

        return bags, torch.tensor(labels)
    
    def save_bag_images(self, bag, bag_labels, label, bag_id):
        fig, axes = plt.subplots(1, len(bag), figsize=(15, 5))
        fig.suptitle(f'Bag {bag_id} - Label: {label}', fontsize=16)

        for i, (img, instance_label) in enumerate(zip(bag, bag_labels)):
            img = img.squeeze(0).numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Label: {instance_label}')
        
        output_path = os.path.join(self.output_bags_dir, f'bag_{bag_id}.png')
        plt.savefig(output_path)
        plt.close(fig)

# Ejemplo de uso de la clase:
if __name__ == "__main__":
    # Crear instancia de la clase BagCreator
    bag_creator = BagCreator(target_digit=3, num_bags=1000, num_instances=10, output_bags_dir='./output/10_inst/bag_creator')
    
    # Crear las bolsas y obtener etiquetas
    bags, labels = bag_creator.create_bags()
