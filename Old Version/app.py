from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import openai
import os

app = Flask(__name__)

# Set your OpenAI API key securely
openai.api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'food_image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['food_image']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        
        # Use GPT-4 to analyze the image and generate a recipe
        recipe = generate_recipe_from_image(image_path)
        
        return render_template('results.html', recipe=recipe)

def generate_recipe_from_image(image_path):
    # Open the image file
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    
    # Use GPT-4 to analyze the image and generate a recipe
    response = openai.Image.create_completion(
        model="gpt-4-vision",
        messages=[
            {"role": "system", "content": "You are a helpful and creative chef."},
            {"role": "user", "content": "Please generate a recipe based on this image of leftover food."}
        ],
        files={"file": image_data}
    )
    
    # Extract the recipe from the response
    recipe = response['choices'][0]['message']['content']
    
    return recipe

if __name__ == '__main__':
    app.run(debug=True)
