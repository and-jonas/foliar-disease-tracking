import os
import numpy as np
import pandas as pd
import glob
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import cv2

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


def get_kpt_bbox(kpts, bbox_sz):

    w = bbox_sz / 2

    x = kpts[:, 0]
    y = kpts[:, 1]

    # get boxes in required format
    bbox = [[x - w, y - w, x + w, y + w] for y, x in zip(y, x)]

    return bbox, (y, x)









base_dir = "Z:/Public/Jonas/Data/ESWW009/SingleLeaf/20240528/JPEG_cam"
image_paths = glob.glob(f"{base_dir}/*.JPG")

# for p in image_paths:
#
#     base_name = os.path.basename(p)
#     name = base_name.replace(".JPG", "")
#
#     img = Image.open(p)
#     img = np.asarray(img)
#     plt.imshow(img)
#
#     output_path = f'{base_dir}/runs/pose/predict/labels/{name}.txt'
#
#     # get key point coordinates from YOLO output
#     coords = pd.read_table(output_path, header=None, sep=" ")
#     x = coords.iloc[:, 5] * 8192
#     x = x.astype("int32")
#     y = coords.iloc[:, 6] * 5464
#     y = y.astype("int32")
#
#     patches = []
#     for i in range(len(x)):
#         p = img[y[i]-28:y[i]+28, x[i]-28:x[i]+28, :]
#         plt.imshow(p)
#

# convert bboxes to min, max
image_path = f'{base_dir}/20240528_094338_ESWW0090023_1.JPG'
output_path = f'{base_dir}/runs/pose/predict/labels/20240528_094338_ESWW0090023_1.txt'


# Load a pre-trained ResNet model and remove the classification head
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fully connected layer

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # Flatten the output


# Initialize the feature extractor
feature_extractor = FeatureExtractor()
feature_extractor.eval()  # Set the model to evaluation mode


# Image preprocessing function
def preprocess_image(image, bbox, input_size=224):
    """
    Extracts and preprocesses an image patch defined by the bounding box.

    :param image: The input image (numpy array).
    :param bbox: The bounding box coordinates (x_min, y_min, x_max, y_max).
    :param input_size: The size to which the patch will be resized (default 224x224).
    :return: Preprocessed image tensor.
    """
    # Extract the patch from the bounding box
    x_min, y_min, x_max, y_max = map(int, bbox)
    patch = image[y_min:y_max, x_min:x_max, :]

    # Convert the patch to a PIL Image
    patch = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))

    # Define transformations: resize, center crop, normalize
    preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transformations to the patch
    patch_tensor = preprocess(patch).unsqueeze(0)  # Add batch dimension
    return patch, patch_tensor


# Function to extract features for each bounding box
def extract_features_for_bboxes(image, bboxes):
    """
    Extracts appearance features for each bounding box in the image.

    :param image: The input image (numpy array).
    :param bboxes: List of bounding boxes (each defined as [x_min, y_min, x_max, y_max]).
    :return: List of feature vectors.
    """
    features = []
    patches = []
    for bbox in bboxes:
        # Preprocess the image patch
        patch, patch_tensor = preprocess_image(image, bbox)
        # Extract features using the CNN
        with torch.no_grad():
            feature_vector = feature_extractor(patch_tensor)

        # Append the feature vector to the list
        features.append(feature_vector.squeeze(0).cpu().numpy())
        patches.append(patch)

    return patches, features


def compute_similarity_matrix(features_frame1, features_frame2):
    """
    Compute the similarity matrix between two sets of feature vectors using cosine similarity.

    :param features_frame1: List of feature vectors from frame 1.
    :param features_frame2: List of feature vectors from frame 2.
    :return: Similarity matrix (2D numpy array).
    """
    # Compute the cosine similarity between feature vectors
    similarity_matrix = cosine_similarity(features_frame1, features_frame2)

    # Convert similarity to distance (if necessary, depending on how you want to use it)
    distance_matrix = 1 - similarity_matrix  # Cosine similarity is between 0 and 1, distance is 1 - similarity

    return distance_matrix


def match_bboxes(features_frame1, features_frame2):
    """
    Match bounding boxes between two frames based on their appearance features.

    :param features_frame1: List of feature vectors from frame 1.
    :param features_frame2: List of feature vectors from frame 2.
    :return: List of matched pairs (index_frame1, index_frame2).
    """
    # Compute the similarity (or distance) matrix
    distance_matrix = compute_similarity_matrix(features_frame1, features_frame2)

    # Apply the Hungarian algorithm to find the optimal matching
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # The result is a list of matched indices between the two frames
    matches = list(zip(row_indices, col_indices))

    return matches


# Example usage

if __name__ == "__main__":

    all_images = glob.glob('Z:/Public/Jonas/Data/ESWW009/SingleLeaf/*/JPEG_cam/*.JPG')

    images = glob.glob("Z:/Public/Jonas/Data/ESWW009/SingleLeaf/Output/ESWW0090023_2/crop/*.JPG")
    images = [os.path.basename(i) for i in images]

    features_images = []
    for frame, i in enumerate(images):

        print(frame)

        image_path = [img for img in all_images if i in img][0]
        image = cv2.imread(image_path)

        img_name = i.replace(".JPG", "")

        # Example bounding boxes (you can replace these with actual YOLOv8 detections)
        # get key point coordinates from YOLO output

        detection_path = f'{os.path.dirname(image_path)}/runs/pose/predict/labels/{img_name}.txt'
        coords = pd.read_table(detection_path, header=None, sep=" ")

        x = coords.iloc[:, 5] * 8192
        x_coords = x.astype("int32")
        y = coords.iloc[:, 6] * 5464
        y_coords = y.astype("int32")

        # get new bbox format
        bboxes = [[x - 28, y - 28, x + 28, y + 28] for y, x in zip(y_coords, x_coords)]
        # Extract features for each bounding box
        patches, features = extract_features_for_bboxes(image, bboxes)

        for id, p in enumerate(patches):
            Path(f"Z:/Public/Jonas/Data/ESWW009/SingleLeaf/Output/ESWW0090023_2/{frame}").mkdir(parents=True, exist_ok=True)
            p.save(f"Z:/Public/Jonas/Data/ESWW009/SingleLeaf/Output/ESWW0090023_2/{frame}/{id}.JPG")
        features_images.append(features)

    matches = []
    for frame in range(1, len(features_images)):
        features_frame1 = features_images[0]
        features_frame2 = features_images[frame]

        # Match the bounding boxes based on their features
        matches.append(match_bboxes(features_frame1, features_frame2))




