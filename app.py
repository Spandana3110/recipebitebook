from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
from sqlalchemy import func
from datetime import datetime
from ml_model.recipe_recommender import RecipeRecommender
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///recipes.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize recipe recommender
recipe_recommender = None
try:
    with app.app_context():
        MODEL_PATH = os.path.join(app.root_path, 'ml_model', 'model.joblib')
        DATASET_PATH = os.path.join(app.root_path, 'data', 'recipes_dataset.csv')
        
        logger.info(f"Loading recipe dataset from {DATASET_PATH}")
        logger.info(f"Loading model from {MODEL_PATH}")
        
        if not os.path.exists(DATASET_PATH):
            logger.error(f"Dataset file not found at {DATASET_PATH}")
            raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")
        
        recipe_recommender = RecipeRecommender(dataset_path=DATASET_PATH, model_path=MODEL_PATH)
    logger.info("Recipe recommender initialized successfully")
except Exception as e:
    logger.error(f"Error initializing recipe recommender: {str(e)}")

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200))
    dietary_preferences = db.Column(db.String(200))
    favorites = db.relationship('Recipe', secondary='favorites')

class Recipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    ingredients = db.Column(db.Text, nullable=False)
    instructions = db.Column(db.Text, nullable=False)
    prep_time = db.Column(db.Integer)
    cook_time = db.Column(db.Integer)
    image_name = db.Column(db.String(200))
    cuisine = db.Column(db.String(50))
    nutritional_info = db.Column(db.Text)

favorites = db.Table('favorites',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id')),
    db.Column('recipe_id', db.Integer, db.ForeignKey('recipe.id'))
)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def get_recipe_features(recipe):
    """Combine recipe features for similarity calculation"""
    features = []
    features.append(recipe.title.lower())
    features.append(recipe.ingredients.lower())
    features.append(recipe.cuisine.lower() if recipe.cuisine else '')
    return ' '.join(features)

def get_similar_recipes(recipe, n=3):
    """Find similar recipes based on content"""
    all_recipes = Recipe.query.all()
    if len(all_recipes) < 2:
        return []
    
    # Create feature vectors
    recipe_features = [get_recipe_features(r) for r in all_recipes]
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(recipe_features)
    
    # Calculate similarity
    cosine_similarities = cosine_similarity(tfidf_matrix)
    
    # Get similar recipes indices
    recipe_idx = all_recipes.index(recipe)
    similar_indices = cosine_similarities[recipe_idx].argsort()[-n-1:-1][::-1]
    
    return [all_recipes[idx] for idx in similar_indices]

def load_recipes_from_csv():
    try:
        # Check if we already have recipes
        if Recipe.query.first() is None:
            df = pd.read_csv('data/recipes_dataset.csv')
            for _, row in df.iterrows():
                recipe = Recipe(
                    title=row['title'],
                    ingredients=row['ingredients'],
                    instructions=row['instructions'] if 'instructions' in row else '',
                    prep_time=row['prep_time'] if 'prep_time' in row else 0,
                    cook_time=row['cook_time'] if 'cook_time' in row else 0,
                    image_name=row['image_name'] if 'image_name' in row else None,
                    cuisine=row['cuisine'] if 'cuisine' in row else None
                )
                db.session.add(recipe)
            db.session.commit()
            print("Recipes loaded successfully!")
    except Exception as e:
        print(f"Error loading recipes: {str(e)}")
        db.session.rollback()

@app.route('/')
def home():
    """Render the home page."""
    try:
        # Get featured recipes
        featured_recipes = [
            {
                'title': 'Mango Cheesecake',
                'description': 'A delightful no-bake cheesecake featuring fresh mango puree and a buttery biscuit base.',
                'Image_Name': 'mangocheesecake.jpg',
                'prep_time': 30,
                'servings': 8,
                'cuisine': 'Dessert'
            },
            {
                'title': 'Tandoori Chicken',
                'description': 'Classic Indian spiced chicken marinated in yogurt and aromatic spices, perfect with naan.',
                'Image_Name': 'chickentandoori.jpg',
                'prep_time': 240,
                'servings': 4,
                'cuisine': 'Indian'
            },
            {
                'title': 'Blue Cocktail',
                'description': 'A refreshing blue lagoon cocktail with vodka, blue curaçao, and lemonade.',
                'Image_Name': 'bluecocktail.jpg',
                'prep_time': 5,
                'servings': 1,
                'cuisine': 'Cocktail'
            }
        ]
        return render_template('home.html', featured_recipes=featured_recipes)
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return "An error occurred", 500

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/search_results', methods=['GET'])
def search_results():
    """Handle recipe search requests."""
    try:
        # Get search parameters
        search_type = request.args.get('search_type', '')
        query = request.args.get('query', '').strip()
        ingredients = request.args.get('ingredients', '').strip()

        logger.info(f"Search request - Type: {search_type}, Query: {query}, Ingredients: {ingredients}")
        
        if not recipe_recommender:
            flash("Recipe recommender is not available", "error")
            return redirect(url_for('search'))

        # Handle empty search
        if not query and not ingredients:
            flash("Please enter a search term", "warning")
            return redirect(url_for('search'))

        recipes = []
        if search_type == 'ingredients' or ingredients:
            # Use ingredients search
            ingredients_list = [ing.strip() for ing in (ingredients or query).split(',') if ing.strip()]
            if not ingredients_list:
                flash("Please enter valid ingredients", "warning")
                return redirect(url_for('search'))
            
            logger.info(f"Searching by ingredients: {ingredients_list}")
            recipes = recipe_recommender.get_recipes_by_ingredients(ingredients_list)
            
        elif search_type == 'recipe_name' or query:
            # Use recipe name search
            logger.info(f"Searching by recipe name: {query}")
            recipes = recipe_recommender.search_recipes_by_name(query)
        
        else:
            flash("Invalid search type", "error")
            return redirect(url_for('search'))

        # Handle search results
        if not recipes:
            search_type_display = "ingredient" if search_type == 'ingredients' or ingredients else "recipe name"
            flash(f"No recipes found for your {search_type_display} search", "info")
            logger.warning(f"No recipes found for {search_type} search with query: {query or ingredients}")
        else:
            logger.info(f"Found {len(recipes)} recipes")
            # Log sample results
            for recipe in recipes[:3]:
                logger.info(f"Sample result - Title: {recipe['title']}, Match: {recipe.get('match_percentage', 'N/A')}%")

        return render_template('search_results.html', 
                             recipes=recipes, 
                             search_type=search_type,
                             query=query or ingredients)

    except Exception as e:
        logger.error(f"Error in search results: {str(e)}")
        flash("An error occurred while searching for recipes", "error")
        return redirect(url_for('search'))

@app.route('/search_by_ingredients', methods=['POST'])
def search_by_ingredients():
    try:
        if recipe_recommender is None:
            flash("Recipe search is currently unavailable. Please try again later.", "error")
            return redirect(url_for('search'))

        ingredients = request.form.get('ingredients', '').strip()
        if not ingredients:
            flash('Please enter at least one ingredient', 'warning')
            return redirect(url_for('search'))

        # Split ingredients and clean them
        ingredient_list = [i.strip() for i in ingredients.split(',') if i.strip()]
        if not ingredient_list:
            flash('Please enter valid ingredients', 'warning')
            return redirect(url_for('search'))
            
        logger.info(f"Searching with ingredients: {ingredient_list}")
        
        # Search for recipes using the recipe recommender
        recipes = recipe_recommender.get_recipes_by_ingredients(ingredient_list)
        
        logger.info(f"Found {len(recipes) if recipes else 0} recipes")
        
        if not recipes:
            flash('No recipes found with those ingredients', 'info')
        
        return render_template('search_results.html',
                             recipes=recipes,
                             search_type='ingredients',
                             query=ingredients)

    except Exception as e:
        logger.error(f"Error in ingredient search: {str(e)}", exc_info=True)
        flash("An error occurred while searching for recipes", "error")
        return redirect(url_for('search'))

@app.route('/recipe/<path:recipe_title>')
def recipe_detail(recipe_title):
    logger.info(f"Accessing recipe: {recipe_title}")
    
    try:
        # URL decode the recipe title (handles double encoding)
        from urllib.parse import unquote
        decoded_title = unquote(unquote(recipe_title))
        logger.info(f"Decoded recipe title: {decoded_title}")
        
        if recipe_recommender is None:
            logger.error("Recipe recommender not initialized")
            flash("Recipe service is currently unavailable", "error")
            return redirect(url_for('home'))
        
        # Try to find the recipe by title
        recipe = recipe_recommender.get_recipe_by_title(decoded_title)
        logger.info(f"Recipe lookup result: {recipe is not None}")
        
        if recipe:
            logger.info(f"Found recipe: {recipe['title']}")
            logger.info(f"Recipe data: {recipe}")
            
            # Ensure recipe has all required fields
            recipe = {
                'title': recipe.get('title', 'Untitled Recipe'),
                'description': recipe.get('description', ''),
                'image_name': recipe.get('image_name', 'default-recipe.jpg'),
                'image_url': url_for('static', filename=f"images/recipes/{recipe.get('image_name', 'default-recipe.jpg')}"),
                'prep_time': recipe.get('prep_time', 30),
                'cook_time': recipe.get('cook_time', 30),
                'servings': recipe.get('servings', 4),
                'cuisine': recipe.get('cuisine', 'International'),
                'ingredients': recipe.get('ingredients', []),
                'instructions': recipe.get('instructions', [])
            }
            
            logger.info(f"Processed recipe data: {recipe}")
            
            try:
                similar_recipes = get_similar_recipes(recipe, n=3)
                logger.info("Found similar recipes")
            except Exception as e:
                logger.error(f"Error finding similar recipes: {str(e)}")
                similar_recipes = []
            
            return render_template('recipe_detail.html', recipe=recipe, similar_recipes=similar_recipes)
        else:
            logger.warning(f"Recipe not found: {decoded_title}")
            flash("Recipe not found", "error")
            return redirect(url_for('search'))
        
    except Exception as e:
        logger.error(f"Error accessing recipe {recipe_title}: {str(e)}", exc_info=True)
        flash("An error occurred while loading the recipe", "error")
        return redirect(url_for('search'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        flash('Invalid email or password', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        dietary_preferences = request.form.get('dietary_preferences')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('signup'))
            
        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            dietary_preferences=dietary_preferences
        )
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash('Account created successfully!', 'success')
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('home'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        try:
            current_user.name = request.form.get('name', current_user.name)
            current_user.dietary_preferences = request.form.get('dietary_preferences', current_user.dietary_preferences)
            
            new_password = request.form.get('new_password')
            if new_password:
                current_user.password_hash = generate_password_hash(new_password)
            
            db.session.commit()
            flash('Profile updated successfully!', 'success')
            
        except Exception as e:
            logger.error(f"Error updating profile: {str(e)}")
            db.session.rollback()
            flash('An error occurred while updating your profile', 'error')
    
    return render_template('profile.html', user=current_user)

@app.route('/favorites')
@login_required
def favorites():
    return render_template('favorites.html', recipes=current_user.favorites)

@app.context_processor
def inject_now():
    return {'now': datetime.now()}

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            load_recipes_from_csv()  # Load recipes if database is empty
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
    app.run(host='0.0.0.0', port=5000, debug=True)
