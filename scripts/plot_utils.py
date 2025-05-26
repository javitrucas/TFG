import cv2
import numpy as np

from matplotlib import pyplot as plt

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def combine_attmap_wsi(attmap, wsi_img):
    """
    Input:
        attmap: numpy array, shape = (H, W)
        wsi_img: numpy array, shape = (H, W, 3), mode RGB
    Output:
        combined_img: numpy array, shape = (H, W, 3), mode RGB
    """
    # attmap = normalize(attmap) # (H, W)
    attmap_img_bgr = cv2.applyColorMap((attmap*255).astype(np.uint8), cv2.COLORMAP_JET) # (H, W, 3)
    wsi_img_bgr = cv2.cvtColor(wsi_img, cv2.COLOR_RGB2BGR) # (H, W, 3)
    combined_img_bgr = cv2.addWeighted(wsi_img_bgr, 0.5, attmap_img_bgr, 0.5, 0) # (H, W, 3)
    combined_img_rgb = cv2.cvtColor(combined_img_bgr, cv2.COLOR_BGR2RGB) # (H, W, 3)
    return combined_img_rgb

def plot_wsi_and_heatmap(
        ax,
        canvas_wsi, 
        attval = None, 
        plot_patch_contour=False,
        size = None,
        row_array = None,
        col_array = None,
        start_y = 0,
        start_x = 0,
        height = None,
        width = None,
        alpha = 0.8 ,
        p = 0.05,
        remove_axis = False
    ):

    if width is None:
        width = canvas_wsi.shape[0]
    if height is None:
        height = canvas_wsi.shape[1]

    # if canvas_attmap is not None:
    #     wsi_array = combine_attmap_wsi(canvas_attmap, canvas_wsi)
    # else:
    #     wsi_array = canvas_wsi
    
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

    canvas_wsi_copy = np.copy(canvas_wsi)

    # ax.imshow(canvas_wsi[start_y:start_y+width, start_x:start_x+height])
    if plot_patch_contour or (attval is not None):  
        for i in range(len(row_array)):
            row_i = row_array[i]
            column_i = col_array[i]
            x_i = column_i * size
            y_i = row_i * size
            row = row_array[i]
            column = col_array[i]
            x = column * size
            y = row * size
            if y_i >= start_y and y_i <= start_y+width and x_i >= start_x and x_i <= start_x+height:
                color = 0
                if attval is not None:
                    w = attval[i]
                    color = 255*(w*tab_red + (1.0-w)*tab_green)
                    # ax.add_patch(plt.Rectangle((x, y), size, size, color=color, alpha=alpha))
                    canvas_wsi_copy[y:y+size, x:x+size] = (alpha)*color + (1.0-alpha)*canvas_wsi[y:y+size, x:x+size]
                    # canvas_wsi_with_heatmap[y:y+size, x:x+size] = 255*color
                
                if plot_patch_contour:
                    contour_len = int(p*size)
                    canvas_wsi_copy[y:y+size, x-contour_len:x+contour_len] = color
                    canvas_wsi_copy[y:y+size, x+size-contour_len:x+size+contour_len] = color
                    
                    canvas_wsi_copy[y-contour_len:y+contour_len, x:x+size] = color
                    canvas_wsi_copy[y+size-contour_len:y+size+contour_len, x:x+size] = color
                    
                    # ax.add_patch(plt.Rectangle((x, y), size, size, edgecolor='black', fill=False))
    ax.imshow(canvas_wsi_copy[start_y:start_y+width, start_x:start_x+height], aspect='equal')
    if remove_axis:
        ax.axis('off')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    # remove axis ticks
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.axis('off')
    return ax

# def plot_wsi_and_heatmap(
#         ax,
#         canvas_wsi, 
#         attval = None, 
#         plot_patch_contour=False,
#         size = None,
#         row_array = None,
#         col_array = None,
#         start_y = 0,
#         start_x = 0,
#         height = None,
#         width = None,
#         alpha = None 
#     ):

#     if width is None:
#         width = canvas_wsi.shape[0]
#     if height is None:
#         height = canvas_wsi.shape[1]
    
#     if alpha is None:
#         alpha = 0.8

#     # if canvas_attmap is not None:
#     #     wsi_array = combine_attmap_wsi(canvas_attmap, canvas_wsi)
#     # else:
#     #     wsi_array = canvas_wsi
    
#     tab_red = np.array([
#         0.8392156862745098,
#         0.15294117647058825,
#         0.1568627450980392
#     ])
#     tab_green = np.array([
#         0.17254901960784313,
#         0.6274509803921569,
#         0.17254901960784313
#     ])

#     ax.imshow(canvas_wsi[start_y:start_y+width, start_x:start_x+height])
#     if plot_patch_contour or (attval is not None):  
#         for i in range(len(row_array)):
#             row_i = row_array[i]
#             column_i = col_array[i]
#             x_i = column_i * size
#             y_i = row_i * size
#             row = row_array[i]
#             column = col_array[i]
#             x = column * size
#             y = row * size
#             if y_i >= start_y and y_i <= start_y+width and x_i >= start_x and x_i <= start_x+height:
#                 if attval is not None:
#                     w = attval[i]
#                     color = w*tab_red + (1.0-w)*tab_green
#                     ax.add_patch(plt.Rectangle((x, y), size, size, color=color, alpha=alpha))
#                 elif plot_patch_contour:
#                     ax.add_patch(plt.Rectangle((x, y), size, size, edgecolor='black', fill=False))
                
#     ax.axis('off')
#     return ax

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
                # ax.add_patch(plt.Rectangle((x, y), size, size, color=color, alpha=alpha))

                canvas_bag_copy[y:y+size, x:x+size] = (alpha)*color + (1.0-alpha)*canvas_bag[y:y+size, x:x+size]

            if plot_patch_contour:
                # ax.add_patch(plt.Rectangle((x, y), size, size, edgecolor='black', fill=False))
                contour_len = int(p*size)
                canvas_bag_copy[y:y+size, x-contour_len:x+contour_len] = color
                canvas_bag_copy[y:y+size, x+size-contour_len:x+size+contour_len] = color

                canvas_bag_copy[y-contour_len:y+contour_len, x:x+size] = color
                canvas_bag_copy[y+size-contour_len:y+size+contour_len, x:x+size] = color

    ax.imshow(canvas_bag_copy[start_y:start_y+width, start_x:start_x+height])
                
    ax.axis('off')
    return ax