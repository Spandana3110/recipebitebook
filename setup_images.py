import os
import shutil
import csv
from pathlib import Path
import re

def clean_filename(name):
    """Clean filename to match standard format."""
    if not isinstance(name, str):
        return ""
    # Convert to lowercase
    name = name.lower()
    # Remove special characters and spaces, replace with underscores
    name = re.sub(r'[^a-z0-9]', '_', name)
    # Remove multiple underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name

def find_appimages_folder():
    """Try to find the appimages folder in common locations."""
    possible_paths = [
        Path.home() / "Desktop" / "appimages",
        Path.home() / "OneDrive" / "Desktop" / "appimages",
        Path.home() / "Downloads" / "appimages",
        Path.home() / "OneDrive" / "Downloads" / "appimages",
        Path.cwd() / "appimages",
        Path.cwd().parent / "appimages",
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"Found appimages folder at: {path}")
            return str(path)
            
    print("Could not find appimages folder in common locations.")
    print("Searched in:")
    for path in possible_paths:
        print(f"  - {path}")
    return None

def setup_recipe_images(source_image_dir, dataset_path):
    """
    Set up recipe images in the correct directory.
    Args:
        source_image_dir: Directory containing your original recipe images
        dataset_path: Path to your recipes_dataset.csv
    """
    # Create target directory if it doesn't exist
    target_dir = Path('static/images/recipes')
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created target directory: {target_dir}")

    # Read dataset
    recipes = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            recipes = list(reader)
        print(f"Found {len(recipes)} recipes in dataset")
        print("Dataset columns:", list(recipes[0].keys()))
    except Exception as e:
        print(f"Error reading dataset: {str(e)}")
        return

    # Process images
    source_dir = Path(source_image_dir)
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        return

    # Get list of all jpeg files
    image_files = list(source_dir.glob('*.jpeg'))
    print(f"Found {len(image_files)} JPEG files in source directory")

    copied_count = 0
    missing_count = 0
    updated_recipes = []

    # For each recipe in the dataset
    for recipe in recipes:
        title = recipe.get('Title', '')  
        clean_title = clean_filename(title)
        
        # Try to find a matching image
        matching_image = None
        for img_file in image_files:
            if clean_filename(img_file.stem) == clean_title:
                matching_image = img_file
                break
        
        if matching_image:
            try:
                # Create new standardized filename
                new_name = f"{clean_title}.jpeg"
                target_path = target_dir / new_name
                
                # Copy the file
                shutil.copy2(matching_image, target_path)
                recipe['Image_Name'] = new_name  
                copied_count += 1
                print(f"Copied: {matching_image.name} -> {new_name}")
            except Exception as e:
                print(f"Error copying {title}: {str(e)}")
                recipe['Image_Name'] = ''  
        else:
            missing_count += 1
            recipe['Image_Name'] = ''  
            print(f"No matching image found for recipe: {title}")
        
        updated_recipes.append(recipe)

    # Update dataset
    try:
        with open(dataset_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=updated_recipes[0].keys())
            writer.writeheader()
            writer.writerows(updated_recipes)
        print("\nUpdated dataset with new image names")
    except Exception as e:
        print(f"Error updating dataset: {str(e)}")

    print("\nSummary:")
    print(f"Total recipes in dataset: {len(recipes)}")
    print(f"Successfully copied: {copied_count}")
    print(f"Missing images: {missing_count}")

if __name__ == "__main__":
    # Try to find the appimages folder
    source_dir = find_appimages_folder()
    if not source_dir:
        print("Please create an 'appimages' folder in one of the above locations and place your recipe images there.")
        exit(1)
        
    dataset_path = 'data/recipes_dataset.csv'
    setup_recipe_images(source_dir, dataset_path)
