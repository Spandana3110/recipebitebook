import os

# Create directory for recipe images
image_path = os.path.join('static', 'images', 'recipes')
os.makedirs(image_path, exist_ok=True)

print(f"Created directory: {image_path}")
