import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np

# Load and preprocess images
def load_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# Load the images
large_image_path = './test/processed_large_image.PNG'
small_image_path = './test/small_image2.PNG'
large_image = load_image(large_image_path)
small_image = load_image(small_image_path)

# Load a pre-trained VGG16 model
model = models.vgg16(pretrained=True).features.eval()

# If GPU is available, move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
large_image = large_image.to(device)
small_image = small_image.to(device)

# Extract feature maps
def get_feature_maps(model, image):
    with torch.no_grad():
        feature_maps = model(image)
    return feature_maps

large_feature_maps = get_feature_maps(model, large_image)
small_feature_maps = get_feature_maps(model, small_image)

# Cross-correlation to find the small image in the large image
result = F.conv2d(large_feature_maps, small_feature_maps)

# Find the maximum correlation value
max_val, max_idx = torch.max(result.view(-1), dim=0)
max_idx = np.unravel_index(max_idx.cpu().numpy(), result.shape[2:])
max_val = max_val.item()

# Calculate the top-left corner of the detected image
x = max_idx[1]
y = max_idx[0]

print(f"Detected location: Top-left corner at ({x}, {y}) with correlation value {max_val}")

# Visualize the result on the large image
def visualize_result(large_image_path, small_image_size, location):
    large_image = Image.open(large_image_path).convert('RGB')
    draw = ImageDraw.Draw(large_image)
    # Draw a thick rectangle around the detected location
    left, top = location
    right, bottom = left + small_image_size[0], top + small_image_size[1] # swapping dimensions for drawing
    draw.rectangle(((left, top), (right, bottom)), outline="red", width=10)
    large_image.show()

# Get size of the small image
small_image_pil = Image.open(small_image_path)
small_image_size = small_image_pil.size

visualize_result(large_image_path, small_image_size, (x, y))
