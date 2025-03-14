import os
import matplotlib.pyplot as plt
import numpy as np

class VisualizationHelper:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_attention_on_image(self, image, patches, attention_weights, bag_id):
        """
        Genera una visualización de la imagen con el heatmap de atención.
        
        Args:
            image: La imagen original completa (numpy array).
            patches: Lista de parches (regiones) de la imagen.
            attention_weights: Pesos de atención para cada parche.
            bag_id: Identificador de la bolsa para nombrar el archivo.
        """
        # Crear una copia de la imagen para superponer el heatmap
        overlay = image.copy()
        
        # Normalizar los pesos de atención para que estén en el rango [0, 1]
        attention_weights = np.array(attention_weights)
        attention_weights = (attention_weights - attention_weights.min()) / (
            attention_weights.max() - attention_weights.min() + 1e-8
        )
        
        # Iterar sobre los parches y superponer los pesos de atención
        for patch, weight in zip(patches, attention_weights):
            # Obtener la posición del parche en la imagen
            x_start, y_start, x_end, y_end = patch['position']  # Ajusta según tu estructura de datos
            
            # Aplicar el peso de atención como un color semitransparente
            overlay[x_start:x_end, y_start:y_end] = (
                overlay[x_start:x_end, y_start:y_end] * (1 - weight) + 
                np.array([255, 0, 0]) * weight  # Rojo para resaltar
            )
        
        # Crear la figura
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title("Imagen Original")
        ax[0].axis('off')
        
        ax[1].imshow(overlay.astype(np.uint8))
        ax[1].set_title("Heatmap de Atención")
        ax[1].axis('off')
        
        # Guardar la figura
        filepath = os.path.join(self.output_dir, f"attention_bag_{bag_id}.png")
        plt.savefig(filepath)
        plt.close(fig)
        print(f"Attention visualization saved at {filepath}")