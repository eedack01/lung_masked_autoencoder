import torch
import numpy as np


def get_class_weights(train_dataset, device, path):
    
    # Extract labels from the train_dataset
    labels = [label for _, label in train_dataset]
    
    # Use np.unique to get unique labels and their counts
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print(label_counts)
    # Calculate total number of samples
    total_samples = len(train_dataset)
    # print(total_samples)
    # min_class_value = np.min(label_counts)
    # Calculate the class weights
    class_weights = {label: total_samples / count for label, count in zip(unique_labels, label_counts)}
    # class_weights = {label: min_class_value / count for label, count in zip(unique_labels, label_counts)}
    # Convert class weights to a tensor
    class_weights_tensor = torch.tensor([class_weights[label] for label in unique_labels], dtype=torch.float32).to(device)
    torch.save(class_weights_tensor, path)
    # normalized_weights = class_weights_tensor / torch.sum(class_weights_tensor)
    return class_weights_tensor