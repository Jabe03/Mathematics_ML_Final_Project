import matplotlib.pyplot as plt
import math
from scipy.linalg import svd
import torch

#SVD functions

def get_SVD_vals(images):
    flattened_images = images.flatten(start_dim=1)
    U, svd_vals, V = svd(flattened_images.T, full_matrices= True)
    return svd_vals

def get_SVD_decomp(images):
    flattened_images = images.flatten(start_dim=1)
    U, svd_vals, V = svd(flattened_images)
    return U, svd_vals, V

def graph_SVDs(svd_vals, labels = None, title = None):
    #print(type(svd_vals))
    if type(svd_vals) is list:
        for svds, label in zip(svd_vals, labels):
            #print("svds", svds)
            x_vals = [i+1 for i in range(len(svds))]
            plt.plot(x_vals, svds, label = label, alpha = 0.5)
    else:
        x_vals = [i+1 for i in range(len(svd_vals))]
        plt.plot(x_vals, svd_vals)
    if title is not None:
        plt.title(title)
    plt.yscale("log")
    plt.legend()
    plt.show()
    
    
    
def recenter(image,new_center):
    width = image[0].size(1)
    height = image[0].size(0)
    original_center = torch.tensor([height/2,  width/2])
    diff = original_center - new_center
    return shift(image, diff)


def find_center_of_brightness(image):
    center = torch.tensor([0.0,0.0])
    for i in range(image.size(1)):
        for j in range(image.size(2)):
            center +=  image[0][i][j]*torch.tensor([j,i])    
    center = center / torch.sum(image[0])
    return center


def shift(image, diff):
    rolled_img = torch.roll(image, (int(diff[0].item()), int(diff[1].item())), dims=(2,1))
    return rolled_img



#Helpful functions

def graph_image(image, label=torch.tensor(0), graph_center = False, graph_point=None):
    plt.imshow(image.squeeze(), cmap="gray")
    if isinstance(label, torch.Tensor):
        plt.title(f"Label: {label.item()}", fontsize=10)
    else:
        plt.title(f"{label}")
    width = image[0].size(1)
    height = image[0].size(0)
    if graph_center:
        plt.plot(width/2, height/2,  marker='o', markersize=5, color='red', label = "Center of image")
    if graph_point is not None:
        plt.plot(graph_point[0], graph_point[1], marker='o', markersize=5, color='blue', label = "Center of brightness")
    if graph_center or graph_point is not None:
        plt.legend()
    plt.show()
    
def graph_images(images,  labels = None, graph_points = None, graph_centers = False, figure_size=100, BATCH_SIZE=256):
    plt.figure(figsize=(figure_size,figure_size))
    for i in range(len(images)):
        image = images[i]
        plt.subplot(int(math.sqrt(BATCH_SIZE)), int(math.sqrt(BATCH_SIZE)), i + 1)
        plt.imshow(image.squeeze(), cmap="gray")
        if labels is not None:
           label = labels[i]
           plt.title(f"{label}", fontsize = figure_size/4)
        width = image[0].size(1)
        height = image[0].size(0)
        if graph_centers:
            plt.plot(width/2, height/2,  marker='o', markersize=figure_size/20, color='red', label = "Center of image")
        if graph_points is not None:
            graph_point = graph_points[i]
            plt.plot(graph_point[0], graph_point[1], marker='o', markersize=5, color='blue', label = "Center of brightness")
            
        if graph_centers or graph_points is not None:
            plt.legend(fontsize = figure_size/10)
        plt.axis("off")
    plt.show()



def generate_box_image(sample_image):
    new_image = torch.zeros_like(sample_image)
    for i in range(new_image.size(1)):
        for j in range(new_image.size(2)):
            if (i == 10 or i == 20) and (j < 20 and j > 10):
                new_image[0][i][j] = 1
            if (j == 10 or j == 20) and (i < 20 and i > 10):
                new_image[0][i][j] = 1
    return new_image

