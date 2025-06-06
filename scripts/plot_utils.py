import cv2
import numpy as np
from matplotlib import pyplot as plt
import os # Added for saving

def normalize(x):
    """Normalizes an array to the range [0, 1]."""
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def combine_attmap_wsi(attmap, wsi_img):
    """
    Combines an attention heatmap with a WSI image using a colormap and alpha blending.
    Input:
        attmap: numpy array, shape = (H, W), attention values (normalized 0-1)
        wsi_img: numpy array, shape = (H, W, 3), mode RGB
    Output:
        combined_img: numpy array, shape = (H, W, 3), mode RGB
    """
    # attmap is already normalized in the main script via `normalize()`
    attmap_img_bgr = cv2.applyColorMap((attmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    wsi_img_bgr = cv2.cvtColor(wsi_img, cv2.COLOR_RGB2BGR)
    combined_img_bgr = cv2.addWeighted(wsi_img_bgr, 0.5, attmap_img_bgr, 0.5, 0)
    combined_img_rgb = cv2.cvtColor(combined_img_bgr, cv2.COLOR_BGR2RGB)
    return combined_img_rgb

# --- THIS IS THE MODIFIED FUNCTION ---
def plot_wsi_and_heatmap(
    wsi_image,              # The WSI thumbnail (numpy array, already loaded)
    attention_scores,       # Normalized attention scores (numpy array, 1D)
    patch_coords,           # Nx2 numpy array of (x, y) coordinates for each patch
    patch_size,             # Original patch size (e.g., 512)
    level_downsample_factor,# Downsampling factor of the `wsi_image` (e.g., 4 for level 2)
    image_id,               # String ID of the WSI
    pred_prob,              # Predicted probability (float)
    true_label,             # True ISUP grade (int or string)
    save_path=None,         # Path to save the image (if not None)
    save_extension='png'    # File extension for saving
):
    """
    Plots a WSI thumbnail with an attention heatmap overlay.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12)) # Create a figure and axis

    # Convert patch coordinates to the thumbnail's coordinate system
    # patch_coords are (x, y) at level 0. Divide by downsample_factor.
    # We are working with (col, row) for x, y in image processing, so patch_coords[:,0] is x (col)
    # and patch_coords[:,1] is y (row)
    
    # Scale patch coordinates and size to the thumbnail's resolution
    scaled_patch_coords = patch_coords / level_downsample_factor
    scaled_patch_size = patch_size / level_downsample_factor

    # Create an empty canvas for the heatmap, same size as the thumbnail
    heatmap_canvas = np.zeros(wsi_image.shape[:2], dtype=np.float32)

    # Populate the heatmap_canvas with attention scores
    # Iterate through each patch and its attention score
    for i in range(len(attention_scores)):
        x_scaled = int(scaled_patch_coords[i, 0])
        y_scaled = int(scaled_patch_coords[i, 1])
        score = attention_scores[i]

        # Define the patch area on the heatmap_canvas
        # Ensure coordinates are within image bounds
        x_end = min(x_scaled + int(scaled_patch_size), heatmap_canvas.shape[1])
        y_end = min(y_scaled + int(scaled_patch_size), heatmap_canvas.shape[0])
        
        # Apply the attention score to the corresponding region in the heatmap_canvas
        # We're just setting a uniform score for the patch area
        # If there are overlapping patches, the last one drawn will overwrite
        if x_scaled < x_end and y_scaled < y_end: # Check for valid dimensions
             heatmap_canvas[y_scaled:y_end, x_scaled:x_end] = score

    # Combine the WSI image and the heatmap
    combined_img = combine_attmap_wsi(heatmap_canvas, wsi_image)

    # Display the combined image
    ax.imshow(combined_img)
    ax.set_title(f"Image ID: {image_id}\nPred. Prob: {pred_prob:.4f} | True Label: {true_label}", fontsize=14)
    ax.axis('off') # Turn off axes for cleaner image

    # Save the plot if save_path is provided
    if save_path:
        filename = f"{image_id}_heatmap.{save_extension}"
        full_save_path = os.path.join(save_path, filename)
        plt.savefig(full_save_path, bbox_inches='tight', dpi=300)
        print(f"Heatmap guardado en: {full_save_path}")
    
    plt.show() # Display the plot


# Keep other functions as they were if you use them elsewhere,
# but plot_scan_and_heatmap is not being called by the main script here.
def plot_scan_and_heatmap(
        ax,
        canvas_bag, 
        attval = None, 
        plot_patch_contour=False,
        size = None,
        alpha = 0.8,
        p = 0.05
    ):

    width = canvas_bag.shape[0]
    height = canvas_bag.shape[1]
    
    tab_red = np.array([
        0.8392156862745098,
        0.15294117647058825,
        0.1568627450980392
    ])
    tab_green = np.array([
        0.17254901960784313,
        0.6274509803921569,
        0.17254901960784313
    ])

    start_x = 0
    start_y = 0

    bag_len = height // size

    canvas_bag_copy = np.copy(canvas_bag)

    if plot_patch_contour or (attval is not None):  
        for i in range(bag_len):
            x = i * size
            y = 0
            color = 0
            if attval is not None:
                w = attval[i]
                color = 255*(w*tab_red + (1.0-w)*tab_green)
                canvas_bag_copy[y:y+size, x:x+size] = (alpha)*color + (1.0-alpha)*canvas_bag[y:y+size, x:x+size]

            if plot_patch_contour:
                contour_len = int(p*size)
                canvas_bag_copy[y:y+size, x-contour_len:x+contour_len] = color
                canvas_bag_copy[y:y+size, x+size-contour_len:x+size+contour_len] = color

                canvas_bag_copy[y-contour_len:y+contour_len, x:x+size] = color
                canvas_bag_copy[y+size-contour_len:y+size+contour_len, x:x+size] = color

    ax.imshow(canvas_bag_copy[start_y:start_y+width, start_x:start_x+height])
                
    ax.axis('off')
    return ax