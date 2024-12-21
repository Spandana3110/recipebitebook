import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import logging
from flask import url_for

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecipeRecommender:
    def __init__(self, dataset_path=None, model_path=None):
        """Initialize the recipe recommender with dataset and model."""
        self.model = None
        self.dataset = None
        self.vectorizer = None
        self.ingredients_matrix = None
        
        # Load the model if available
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.warning(f"Error loading model: {str(e)}")
        
        # Load and prepare the dataset
        if not dataset_path or not os.path.exists(dataset_path):
            raise ValueError("Valid dataset path is required")
            
        try:
            logger.info(f"Loading dataset from {dataset_path}")
            self.dataset = pd.read_csv(dataset_path, encoding='utf-8')
            logger.info(f"Dataset loaded successfully with {len(self.dataset)} recipes")
            
            # Clean and prepare the dataset
            self._prepare_dataset()
            
            # Create TF-IDF vectorizer for ingredients
            self._prepare_ingredient_matching()
            
            logger.info("Recipe recommender initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing recipe recommender: {str(e)}")
            raise

    def _prepare_dataset(self):
        """Clean and prepare the dataset."""
        try:
            # Add featured recipes if not already in dataset
            featured_recipes = [
                {
                    'Title': 'Mango Cheesecake',
                    'Ingredients': 'digestive biscuits, unsalted butter, cream cheese, mango puree, sugar, vanilla extract, whipping cream, gelatin, water',
                    'Instructions': 'Mix crushed biscuits with melted butter and press into pan. Chill. Dissolve gelatin in water and heat. Beat cream cheese, sugar, and vanilla. Add mango puree. Fold in whipped cream. Mix in gelatin and pour over crust. Refrigerate. Add mango topping and chill.',
                    'PrepTime': 30,
                    'CookTime': 240,
                    'Servings': 8,
                    'Cuisine': 'Dessert',
                    'Image_Name': 'mangocheesecake.jpg'
                },
                {
                    'Title': 'Tandoori Chicken',
                    'Ingredients': 'chicken, yogurt, lemon juice, ginger-garlic paste, turmeric powder, red chili powder, cumin powder, coriander powder, garam masala, tandoori masala, mustard oil, salt',
                    'Instructions': 'Mix marinade ingredients. Make slits in chicken and marinate for 4 hours. Preheat oven to 200°C. Roast for 25-30 minutes, flipping halfway. Garnish with coriander and serve.',
                    'PrepTime': 240,
                    'CookTime': 30,
                    'Servings': 4,
                    'Cuisine': 'Indian',
                    'Image_Name': 'chickentandoori.jpg'
                },
                {
                    'Title': 'Blue Cocktail',
                    'Ingredients': 'vodka, blue curaçao, lemonade, ice cubes, lemon',
                    'Instructions': 'Fill glass with ice. Pour vodka and blue curaçao. Top with lemonade. Stir and garnish with lemon.',
                    'PrepTime': 5,
                    'CookTime': 0,
                    'Servings': 1,
                    'Cuisine': 'Cocktail',
                    'Image_Name': 'bluecocktail.jpg'
                }
            ]
            
            # Add featured recipes to dataset if they don't exist
            for recipe in featured_recipes:
                if recipe['Title'] not in self.dataset['Title'].values:
                    self.dataset = pd.concat([self.dataset, pd.DataFrame([recipe])], ignore_index=True)
            
            # Clean the dataset
            self.dataset.fillna('', inplace=True)
            self.dataset['Title'] = self.dataset['Title'].astype(str)
            self.dataset['Ingredients'] = self.dataset['Ingredients'].astype(str)
            self.dataset['Instructions'] = self.dataset['Instructions'].astype(str)
            
            # Infer cuisine type if missing
            def infer_cuisine(row):
                ingredients = str(row['Ingredients']).lower()
                title = str(row['Title']).lower()
                cuisine = str(row.get('Cuisine', '')).strip()
                
                if not cuisine or cuisine == 'nan':
                    # Indian cuisine indicators
                    indian_ingredients = ['garam masala', 'curry', 'turmeric', 'cumin', 'coriander', 'tandoori', 'paneer', 'masala']
                    if any(ing in ingredients for ing in indian_ingredients):
                        return 'Indian'
                    
                    # Italian cuisine indicators
                    italian_ingredients = ['pasta', 'pizza', 'mozzarella', 'parmesan', 'basil', 'oregano']
                    if any(ing in ingredients for ing in italian_ingredients):
                        return 'Italian'
                    
                    # Chinese cuisine indicators
                    chinese_ingredients = ['soy sauce', 'sesame oil', 'ginger', 'tofu', 'noodles']
                    if any(ing in ingredients for ing in chinese_ingredients):
                        return 'Chinese'
                    
                    # Mexican cuisine indicators
                    mexican_ingredients = ['tortilla', 'taco', 'cilantro', 'jalapeno', 'enchilada']
                    if any(ing in ingredients for ing in mexican_ingredients):
                        return 'Mexican'
                    
                    # Dessert indicators
                    dessert_ingredients = ['sugar', 'chocolate', 'vanilla', 'cream', 'butter', 'cake', 'cookie']
                    if any(ing in ingredients for ing in dessert_ingredients) or any(word in title for word in ['cake', 'cookie', 'dessert', 'sweet', 'pie']):
                        return 'Dessert'
                    
                    # Beverage indicators
                    beverage_ingredients = ['vodka', 'rum', 'whiskey', 'juice', 'cocktail', 'wine']
                    if any(ing in ingredients for ing in beverage_ingredients) or 'drink' in title or 'cocktail' in title:
                        return 'Beverage'
                    
                    return 'International'
                return cuisine
            
            # Apply cuisine inference
            self.dataset['Cuisine'] = self.dataset.apply(infer_cuisine, axis=1)
            
            logger.info(f"Dataset prepared successfully with {len(self.dataset)} recipes")
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            raise

    def _prepare_ingredient_matching(self):
        """Prepare the ingredient matching system."""
        try:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.ingredients_matrix = self.vectorizer.fit_transform(self.dataset['Ingredients'])
            logger.info("Ingredient matching system prepared successfully")
            
        except Exception as e:
            logger.error(f"Error preparing ingredient matching: {str(e)}")
            raise

    def _safe_int_convert(self, value, default):
        """Safely convert a value to integer, returning default if conversion fails."""
        try:
            if pd.isna(value) or value == '':
                return default
            return int(float(value))
        except (ValueError, TypeError):
            return default

    def get_recipes_by_ingredients(self, ingredients_list, num_recipes=5):
        """Get recipe recommendations based on input ingredients."""
        try:
            if not ingredients_list:
                logger.error("Empty ingredients list")
                return []
            
            # Clean and prepare ingredients list
            ingredients_query = ', '.join(ingredients_list)
            logger.info(f"Searching for recipes with ingredients: {ingredients_query}")
            
            # Search for recipes containing any of the ingredients
            matching_recipes = pd.DataFrame()
            for ingredient in ingredients_list:
                ingredient = ingredient.lower().strip()
                matches = self.dataset[
                    self.dataset['Ingredients'].str.lower().str.contains(ingredient, na=False)
                ]
                matching_recipes = pd.concat([matching_recipes, matches]).drop_duplicates()
            
            if matching_recipes.empty:
                logger.warning(f"No recipes found with ingredients: {ingredients_query}")
                return []
            
            # Calculate match percentage based on number of matching ingredients
            matching_recipes['match_percentage'] = matching_recipes['Ingredients'].apply(
                lambda x: sum(1 for ing in ingredients_list if ing.lower() in str(x).lower()) * 100 / len(ingredients_list)
            )
            
            # Sort by match percentage and get top matches
            matching_recipes = matching_recipes.sort_values('match_percentage', ascending=False)
            top_matches = matching_recipes.head(num_recipes)
            
            # Convert to list of dictionaries with safe numeric conversions
            results = []
            for _, recipe in top_matches.iterrows():
                try:
                    result = {
                        'title': str(recipe.get('Title', '')),
                        'ingredients': [ing.strip() for ing in str(recipe.get('Ingredients', '')).split(',') if ing.strip()],
                        'instructions': [inst.strip() for inst in str(recipe.get('Instructions', '')).split('.') if inst.strip()],
                        'prep_time': self._safe_int_convert(recipe.get('PrepTime'), 30),
                        'cook_time': self._safe_int_convert(recipe.get('CookTime'), 30),
                        'servings': self._safe_int_convert(recipe.get('Servings'), 4),
                        'cuisine': str(recipe.get('Cuisine', 'International')).strip() or 'International',
                        'image_name': str(recipe.get('Image_Name', 'default-recipe.jpg')),
                        'match_percentage': float(recipe.get('match_percentage', 0))
                    }
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing recipe {recipe.get('Title', '')}: {str(e)}")
                    continue
            
            logger.info(f"Found {len(results)} recipes")
            return results
            
        except Exception as e:
            logger.error(f"Error in recipe recommendation: {str(e)}")
            return []

    def search_recipes_by_name(self, query, max_results=10):
        """Search recipes by name."""
        try:
            if not query:
                logger.error("Empty search query")
                return []
            
            query = query.lower().strip()
            logger.info(f"Searching for recipes with name: {query}")
            
            # Search for recipes containing the query in the title
            matching_recipes = self.dataset[
                self.dataset['Title'].str.lower().str.contains(query, na=False)
            ]
            
            # Sort by relevance (exact matches first, then partial matches)
            matching_recipes['match_percentage'] = matching_recipes['Title'].apply(
                lambda x: 100 if x.lower() == query else 85
            )
            
            # Get top matches
            top_matches = matching_recipes.head(max_results)
            
            # Convert to list of dictionaries with safe numeric conversions
            results = []
            for _, recipe in top_matches.iterrows():
                try:
                    result = {
                        'title': str(recipe.get('Title', '')),
                        'ingredients': [ing.strip() for ing in str(recipe.get('Ingredients', '')).split(',') if ing.strip()],
                        'instructions': [inst.strip() for inst in str(recipe.get('Instructions', '')).split('.') if inst.strip()],
                        'prep_time': self._safe_int_convert(recipe.get('PrepTime'), 30),
                        'cook_time': self._safe_int_convert(recipe.get('CookTime'), 30),
                        'servings': self._safe_int_convert(recipe.get('Servings'), 4),
                        'cuisine': str(recipe.get('Cuisine', 'International')).strip() or 'International',
                        'image_name': str(recipe.get('Image_Name', 'default-recipe.jpg')),
                        'match_percentage': float(recipe.get('match_percentage', 0))
                    }
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing recipe {recipe.get('Title', '')}: {str(e)}")
                    continue
            
            logger.info(f"Found {len(results)} recipes matching '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error in recipe search: {str(e)}")
            return []

    def get_recipe_by_id(self, recipe_id):
        """Get a specific recipe by ID."""
        try:
            if not isinstance(recipe_id, int) or recipe_id < 4:
                logger.error(f"Invalid recipe ID: {recipe_id}")
                return None
            
            # Adjust recipe_id to account for featured recipes offset
            dataset_idx = recipe_id - 4
            
            if dataset_idx >= len(self.dataset):
                logger.error(f"Recipe ID {recipe_id} out of range")
                return None
            
            recipe = self.dataset.iloc[dataset_idx]
            logger.info(f"Found recipe: {recipe.get('Title', 'Unknown')}")
            
            # Convert recipe data to dictionary with consistent format
            recipe_dict = {
                'id': recipe_id,
                'title': recipe.get('Title', 'Untitled Recipe'),
                'description': recipe.get('Description', ''),
                'ingredients': recipe.get('Ingredients', ''),
                'instructions': recipe.get('Instructions', ''),
                'prep_time': self._safe_int_convert(recipe.get('PrepTime'), 30),
                'cook_time': self._safe_int_convert(recipe.get('CookTime'), 30),
                'servings': self._safe_int_convert(recipe.get('Servings'), 4),
                'cuisine': recipe.get('Cuisine', 'International'),
                'image_name': recipe.get('Image_Name', 'default-recipe.jpg'),
                'image_url': url_for('static', filename=f"images/recipes/{recipe.get('Image_Name', 'default-recipe.jpg')}")
            }
            
            return recipe_dict
            
        except Exception as e:
            logger.error(f"Error getting recipe by ID {recipe_id}: {str(e)}")
            return None

    def get_recipe_by_title(self, title):
        """Get a recipe by its title."""
        try:
            if not title:
                logger.error("Empty title provided")
                return None
            
            # Find the recipe with matching title (case-insensitive)
            search_title = title.lower().strip()
            logger.info(f"Searching for recipe with title: {search_title}")
            
            # Use pandas filtering for efficiency
            mask = self.dataset['Title'].str.lower().str.strip() == search_title
            matching_recipes = self.dataset[mask]
            
            if not matching_recipes.empty:
                recipe = matching_recipes.iloc[0]
                logger.info(f"Found recipe: {recipe['Title']}")
                
                # Convert ingredients and instructions to lists and clean them
                ingredients = [ing.strip() for ing in str(recipe.get('Ingredients', '')).split(',') if ing.strip()]
                instructions = [inst.strip() for inst in str(recipe.get('Instructions', '')).split('.') if inst.strip()]
                
                # Create recipe dictionary with all necessary fields
                recipe_dict = {
                    'title': recipe.get('Title', 'Untitled Recipe'),  # Use original title case
                    'description': recipe.get('Description', ''),  # Get description if available
                    'ingredients': ingredients,
                    'instructions': instructions,
                    'prep_time': self._safe_int_convert(recipe.get('PrepTime'), 30),
                    'cook_time': self._safe_int_convert(recipe.get('CookTime'), 30),
                    'servings': self._safe_int_convert(recipe.get('Servings'), 4),
                    'cuisine': recipe.get('Cuisine', 'International'),  # Try to get actual cuisine
                    'image_name': recipe.get('Image_Name', 'default-recipe.jpg')
                }
                
                logger.info(f"Processed recipe data: {recipe_dict}")
                return recipe_dict
            
            # If exact match not found, try partial match
            mask = self.dataset['Title'].str.lower().str.strip().str.contains(search_title, regex=False)
            matching_recipes = self.dataset[mask]
            
            if not matching_recipes.empty:
                recipe = matching_recipes.iloc[0]
                logger.info(f"Found recipe with partial match: {recipe['Title']}")
                
                # Convert ingredients and instructions to lists and clean them
                ingredients = [ing.strip() for ing in str(recipe.get('Ingredients', '')).split(',') if ing.strip()]
                instructions = [inst.strip() for inst in str(recipe.get('Instructions', '')).split('.') if inst.strip()]
                
                # Create recipe dictionary with all necessary fields
                recipe_dict = {
                    'title': recipe.get('Title', 'Untitled Recipe'),  # Use original title case
                    'description': recipe.get('Description', ''),  # Get description if available
                    'ingredients': ingredients,
                    'instructions': instructions,
                    'prep_time': self._safe_int_convert(recipe.get('PrepTime'), 30),
                    'cook_time': self._safe_int_convert(recipe.get('CookTime'), 30),
                    'servings': self._safe_int_convert(recipe.get('Servings'), 4),
                    'cuisine': recipe.get('Cuisine', 'International'),  # Try to get actual cuisine
                    'image_name': recipe.get('Image_Name', 'default-recipe.jpg')
                }
                
                logger.info(f"Processed recipe data: {recipe_dict}")
                return recipe_dict
                
            logger.warning(f"No recipe found for title: {title}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting recipe by title {title}: {str(e)}")
            return None
