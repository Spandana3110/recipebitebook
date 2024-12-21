// Ingredient Search Form Handling
document.addEventListener('DOMContentLoaded', function() {
    const ingredientForm = document.getElementById('ingredient-search');
    if (ingredientForm) {
        const ingredientsContainer = ingredientForm.querySelector('.ingredients-container');
        const addButton = ingredientsContainer.querySelector('button');

        // Add new ingredient input
        addButton.addEventListener('click', function() {
            const newInput = document.createElement('div');
            newInput.className = 'input-group mb-2';
            newInput.innerHTML = `
                <input type="text" class="form-control" placeholder="Add an ingredient">
                <button class="btn btn-outline-danger" type="button">-</button>
            `;
            ingredientsContainer.appendChild(newInput);

            // Add remove button functionality
            newInput.querySelector('button').addEventListener('click', function() {
                newInput.remove();
            });
        });

        // Form submission
        ingredientForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const ingredients = Array.from(ingredientForm.querySelectorAll('input'))
                                   .map(input => input.value.trim())
                                   .filter(value => value !== '');
            
            if (ingredients.length > 0) {
                // Send ingredients to backend
                fetch('/search-by-ingredients', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ ingredients: ingredients })
                })
                .then(response => response.json())
                .then(data => {
                    // Handle the response
                    window.location.href = '/search-results?ingredients=' + 
                        encodeURIComponent(ingredients.join(','));
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            }
        });
    }
});

// Favorite Recipe Handling
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('favorite-btn')) {
        const recipeId = e.target.dataset.recipeId;
        
        fetch('/toggle-favorite/' + recipeId, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                e.target.classList.toggle('active');
                const icon = e.target.querySelector('i');
                if (icon) {
                    icon.classList.toggle('fas');
                    icon.classList.toggle('far');
                }
            }
        })
        .catch(error => console.error('Error:', error));
    }
});

// Image Preview for Recipe Upload
const imageInput = document.getElementById('recipe-image');
if (imageInput) {
    imageInput.addEventListener('change', function(e) {
        const preview = document.getElementById('image-preview');
        const file = e.target.files[0];
        
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            }
            reader.readAsDataURL(file);
        }
    });
}

// Form Validation
const forms = document.querySelectorAll('.needs-validation');
forms.forEach(form => {
    form.addEventListener('submit', function(e) {
        if (!form.checkValidity()) {
            e.preventDefault();
            e.stopPropagation();
        }
        form.classList.add('was-validated');
    });
});
