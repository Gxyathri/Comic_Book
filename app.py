#improving prompt

import pandas as pd
import requests
import google.generativeai as genai
import io
import os
import time
import re
from flask import Flask, render_template, request, jsonify, redirect, flash, send_file
import mysql.connector
import string
from diffusers import StableDiffusionPipeline
from diffusers import models
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from gtts import gTTS
from moviepy.editor import *
from fpdf import FPDF
from pathlib import Path


app = Flask(__name__)

# Configure API keys
genai.configure(api_key="replace with your key")
STABLE_DIFFUSION_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
STABLE_DIFFUSION_API_KEY = "replace with your key"

def fetch_story_from_excel(image_id):
    # Load the Excel file
    excel_file_path = 'C:/Users/gayat/OneDrive/Desktop/comic/Stories for PixelPlot.xlsx'  
    df = pd.read_excel(excel_file_path)
    
    # Assuming the Excel file has columns 'Image_ID' and 'Story', you would filter based on 'Image_ID'
    filtered_df = df[df['Image_ID'] == image_id]
    
    # Assuming there's only one story per image_id, you can return the first result
    if not filtered_df.empty:
        story = filtered_df.iloc[0]['Story']
        return story
    else:
        # Handle case where no story is found for the given image_id
        return "Story not found"


def clear_folder(folder_path):
    # List all files and directories in the given folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Check if the path is a file
        if os.path.isfile(file_path):
            # Remove the file
            os.remove(file_path)
        elif os.path.isdir(file_path):
            # Recursively clear the subdirectory
            clear_folder(file_path)
            # Remove the directory
            os.rmdir(file_path)

def generate_sentence_prompts(summary):
    """Generate prompts for each sentence while maintaining full context"""
    base_style = "cartoon style, children's book illustration, vibrant colors, whimsical, disney-inspired, "
    
    # Split summary into sentences and clean them
    sentences = [s.strip() for s in summary.split('.') if s.strip()]
    
    # Generate a prompt for each sentence with full context
    prompts = []
    for sentence in sentences:
        prompt = f"{base_style} illustrate this scene: {sentence}. Full story context: {summary}"
        prompts.append(prompt)
    
    return prompts, sentences

def generate_image(prompt, max_retries=5, initial_retry_delay=60):
    headers = {
        "Authorization": f"Bearer {STABLE_DIFFUSION_API_KEY}"
    }
    
    payload = {
        "inputs": prompt,
        "options": {
            "wait_for_model": True
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(STABLE_DIFFUSION_API_URL, headers=headers, json=payload)
            
            if response.status_code == 429:
                retry_delay = initial_retry_delay * (attempt + 1)
                print(f"Rate limit reached. Waiting {retry_delay} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
                continue
            
            elif response.status_code == 503:
                print(f"Model loading, attempt {attempt + 1} of {max_retries}")
                time.sleep(10)
                continue
            
            elif response.status_code == 200:
                if len(response.content) > 100:
                    try:
                        Image.open(io.BytesIO(response.content))
                        return response.content
                    except Exception as e:
                        print(f"Received invalid image data: {e}")
                else:
                    print("Received response too small to be valid image")
            else:
                print(f"Error: Status Code {response.status_code}")
                print(f"Response: {response.text}")
            
            time.sleep(10)
            
        except Exception as e:
            print(f"Exception during attempt {attempt + 1}: {e}")
            time.sleep(10)
    
    print(f"Failed to generate image after {max_retries} attempts")
    return None

def process_story_images(prompts_and_sentences, batch_size=3, batch_delay=60):
    """Process story-based image prompts in batches"""
    prompts, sentences = prompts_and_sentences
    image_results = []
    total_items = len(prompts)
    
    for i in range(0, total_items, batch_size):
        # Calculate the end index for current batch
        batch_end = min(i + batch_size, total_items)
        current_prompts = prompts[i:batch_end]
        current_sentences = sentences[i:batch_end]
        
        print(f"Processing batch {(i // batch_size) + 1}")
        
        # Process each prompt in the current batch
        for prompt, sentence in zip(current_prompts, current_sentences):
            print(f"Generating image for sentence: {sentence}")
            image_bytes = generate_image(prompt)
            
            if image_bytes:
                image_path = save_image(image_bytes)
                if image_path:
                    image_results.append({
                        'sentence': sentence,
                        'image_path': image_path
                    })
                else:
                    image_results.append({
                        'sentence': sentence,
                        'image_path': "Error saving image"
                    })
            else:
                image_results.append({
                    'sentence': sentence,
                    'image_path': "Error generating image"
                })
        
        # If there are more batches to process, wait
        if batch_end < total_items:
            print(f"Waiting {batch_delay} seconds before processing next batch...")
            time.sleep(batch_delay)
    
    return image_results
    
def save_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image_folder = "generated_images"
        os.makedirs(image_folder, exist_ok=True)
        image_path = os.path.join(image_folder, f"story_image_{int(time.time())}.png")
        image.save(image_path)
        return image_path
    except Exception as e:
        print(f"Error saving image: {e}")
        return None
app.secret_key = 'niwdyhwbec' # Replace with a secret key for flashing messages
user_logged_in=False

@app.route("/")
def signin_or_index():
    if user_logged_in:  
        return redirect('/index')
    else:
        return redirect('/signin')
    
@app.route("/signin", methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='password',
            db='comic'
        )
        cursor = conn.cursor()

        query = "SELECT * FROM customer WHERE password = %s AND email = %s"
        values = (password, email)

        cursor.execute(query, values)
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user:
            # Set a variable or session indicating that the user is logged in
            # For simplicity, let's assume a variable named 'user_logged_in'
            user_logged_in = True
            return redirect('/index')
        else:
            flash('ACCOUNT DOES NOT EXIST', 'error')

    return render_template('signin.html')

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='password',
            db='comic'
        )

        if (".in" in email or ".com" in email) and len(password) > 7:

            if any(char in string.punctuation for char in password) != True:
                flash('INCLUDE SPECIAL CHARACTER IN PASSWORD', 'error')

            else:

                cursor = conn.cursor()

                query = "SELECT email FROM customer WHERE password = %s AND email = %s"
                values = (password, email)

                cursor.execute(query, values)
                user = cursor.fetchone()

                cursor.close()
                conn.close()

                if user:
                    flash('ACCOUNT ALREADY EXISTS', 'error')

                else:
                    conn = mysql.connector.connect(
                        host='localhost',
                        user='root',
                        password='password',
                        db='dbname'
                    )
                    cursor = conn.cursor()
                    query = "INSERT INTO customer (password, email) VALUES (%s, %s)"
                    values = (password, email)

                    cursor.execute(query, values)
                    conn.commit()

                    cursor.close()
                    conn.close()

        else:
            flash('INVALID EMAIL OR PASSWORD', 'error')

    return render_template('signup.html')
def generate_pdf_with_images(images_folder):
    pdf = FPDF()
    image_files = [f for f in os.listdir("C:/Users/gayat/OneDrive/Desktop/comic/static/image") if f.endswith('.png')]
    for image_file in image_files:
        pdf.add_page()
        pdf.image(os.path.join(images_folder, image_file), x=10, y=10, w=190)
    pdf_file = 'C:/Users/gayat/OneDrive/Desktop/comic/flipbook.pdf'
    pdf.output(pdf_file)
    return pdf_file

def insert_line_breaks(text, max_length):
    words = text.split()
    lines = []
    current_line = ''
    for word in words:
        if len(current_line + word) <= max_length:
            current_line += word + ' '
        else:
            lines.append(current_line.strip())
            current_line = word + ' '
    if current_line:
        lines.append(current_line.strip())
    return '\n'.join(lines)

@app.route('/book1')
def book1():
    return render_template('book1.html')
    
@app.route('/index', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Initialize lists
        processed_pages = []
        clips = []
        
        # Clear folders
        clear_folder('static/image')
        clear_folder('static/video')
        clear_folder('static/audio')
        clear_folder('C:/Users/gayat/OneDrive/Desktop/comic/generated_images')
        
        # Get user input from form
        story = request.form.get('textInput')
        
        try:
            # If story is longer than 200 words, summarize it
            word_count = len(story.split())
            if word_count > 50:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(f"Summarize the following story to 5 sentences: {story}")
                summary = response.text if response and hasattr(response, 'text') and response.text else story
            else:
                summary = story
            
            # Generate prompts and process images
            prompts_and_sentences = generate_sentence_prompts(summary)
            image_results = process_story_images(prompts_and_sentences)
            
            # Process the summary into formatted sentences
            sentences = [s.strip() for s in summary.split('.') if s.strip()]
            formatted_sentences = [insert_line_breaks(sentence, max_length=50) for sentence in sentences]
            
            # Get and sort image files
            image_files = sorted(Path('C:/Users/gayat/OneDrive/Desktop/comic/generated_images').glob('*.[pj][np][g]'))
            
            # Process each image and its corresponding text
            for index, (formatted_text, image_file) in enumerate(zip(formatted_sentences, image_files)):
                if image_file.exists():
                    processed_pages.append({
                        'image_path': f'/static/image/{index + 1}.png',
                        'text': formatted_text
                    })
                    
                    # Process image
                    with Image.open(image_file) as img:
                        img = img.resize((1024, 1024)) if img.size != (1024, 1024) else img
                        draw = ImageDraw.Draw(img)
                        font = ImageFont.truetype("C:/Users/gayat/OneDrive/Desktop/comic/PlayfairDisplay-Regular.otf", 28)
                        
                        # Calculate text dimensions
                        lines = formatted_text.split('\n')
                        sample_bbox = font.getbbox("Ag")
                        line_height = (sample_bbox[3] - sample_bbox[1]) + 5
                        total_text_height = line_height * len(lines)
                        
                        # Draw text background
                        rect_height = total_text_height + 40
                        rect_y = 1024 - rect_height - 20
                        draw.rectangle([(0, rect_y), (1024, rect_y + rect_height)], fill="white")
                        
                        # Draw text
                        current_y = rect_y + 20
                        for line in lines:
                            bbox = font.getbbox(line)
                            line_width = bbox[2] - bbox[0]
                            line_x = (1024 - line_width) // 2
                            draw.text((line_x, current_y), line, font=font, fill="black")
                            current_y += line_height
                        
                        # Save processed image
                        output_path = Path('static/image') / f"{index + 1}.png"
                        img.save(output_path)
                        
                        # Generate audio
                        tts = gTTS(text=formatted_text, lang='en', slow=False)
                        audio_path = f"static/audio/voicecover{index + 1}.mp3"
                        tts.save(audio_path)
                        
                        # Create video clip
                        audio_clip = AudioFileClip(audio_path)
                        image_clip = ImageClip(str(output_path)).set_duration(audio_clip.duration)
                        video_clip = image_clip.set_audio(audio_clip)
                        clips.append(video_clip)
            
            # Create final video
            if clips:
                final_video = concatenate_videoclips(clips, method="compose")
                final_video.write_videofile("static/final_video1.mp4", fps=24)
            
            return render_template('book1.html', pages=processed_pages)
            
        except Exception as e:
            print(f"Error in home endpoint: {e}")
            return jsonify({"error": str(e)}), 500
            
    # If GET request, show the input form
    return render_template('index.html')


@app.route('/get_story', methods=['GET','POST'])
def get_story():
    # Initialize the processed_pages list
    processed_pages = []
    clips = [] 
    
    # Clear all necessary folders
    clear_folder('static/image')
    clear_folder('static/video')
    clear_folder('static/audio')
    clear_folder('C:/Users/gayat/OneDrive/Desktop/comic/generated_images')
    
    # Get the image ID from request parameters
    image_id = request.args.get('image_id')
    story = fetch_story_from_excel(image_id)
    
    try:
        # Generate the summary using Google Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Summarize the following story to 5 sentences: {story}")
        
        # Check if we have a valid response
        if response and hasattr(response, 'text') and response.text:
            summary = response.text
        else:
            summary = "No valid content generated"
        
        # Generate prompts and process images
        prompts_and_sentences = generate_sentence_prompts(summary)
        image_results = process_story_images(prompts_and_sentences)
        
        # Process the summary into formatted sentences
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        formatted_sentences = [insert_line_breaks(sentence, max_length=50) for sentence in sentences]
        
        # Get and sort image files
        image_files = sorted(Path('C:/Users/gayat/OneDrive/Desktop/comic/generated_images').glob('*.[pj][np][g]'))
        
        # Process each image and its corresponding text
        for index, (formatted_text, image_file) in enumerate(zip(formatted_sentences, image_files)):
            if image_file.exists():
                # Add page information to our list
                processed_pages.append({
                    'image_path': f'/static/image/{index + 1}.png',
                    'text': formatted_text
                })
                
                # Open and process the image
                with Image.open(image_file) as img:
                    # Resize if necessary
                    if img.size != (1024, 1024):
                        img = img.resize((1024, 1024))
                    
                    # Prepare for drawing
                    draw = ImageDraw.Draw(img)
                    font = ImageFont.truetype("C:/Users/gayat/OneDrive/Desktop/comic/PlayfairDisplay-Regular.otf", 28)
                    
                    # Calculate text dimensions using getbbox() instead of getsize()
                    lines = formatted_text.split('\n')
                    # Calculate line height using a sample text's bounding box
                    sample_bbox = font.getbbox("Ag")
                    line_height = (sample_bbox[3] - sample_bbox[1]) + 5  # Adding 5 pixels padding
                    total_text_height = line_height * len(lines)
                    
                    # Draw white background rectangle
                    rect_height = total_text_height + 40  # 40 pixels padding (20 top + 20 bottom)
                    rect_y = 1024 - rect_height - 20
                    draw.rectangle([(0, rect_y), (1024, rect_y + rect_height)], fill="white")
                    
                    # Draw text lines
                    current_y = rect_y + 20  # Start 20 pixels from the top of the rectangle
                    for line in lines:
                        # Get the bounding box for the current line
                        bbox = font.getbbox(line)
                        line_width = bbox[2] - bbox[0]  # Right minus left gives width
                        # Center the text horizontally
                        line_x = (1024 - line_width) // 2
                        draw.text((line_x, current_y), line, font=font, fill="black")
                        current_y += line_height
                    
                    # Save the processed image
                    output_path = Path('static/image') / f"{index + 1}.png"
                    img.save(output_path)
                    tts = gTTS(text=formatted_text, lang='en', slow=False)
                    audio_path = f"static/audio/voicecover{index + 1}.mp3"
                    tts.save(audio_path)

                    # Create video clip
                    audio_clip = AudioFileClip(audio_path)
                    image_clip = ImageClip(str(output_path)).set_duration(audio_clip.duration)
                    video_clip = image_clip.set_audio(audio_clip)
                    clips.append(video_clip)

        if clips:
            final_video = concatenate_videoclips(clips, method="compose")
            final_video.write_videofile("static/final_video1.mp4", fps=24)                
                    
        
    except Exception as e:
        print(f"Error in get story endpoint: {e}")
        return jsonify({"error": str(e)}), 500
    
        
    
    # Return the template with the processed pages
    return render_template('book1.html', pages=processed_pages)

@app.route('/browse')
def browse():
    return render_template('browse.html')

@app.route('/video')
def video():
    video_path = "static/final_video1.mp4"  # Replace with the actual path to your video file
    return render_template('video.html', video_path=video_path)

@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    pdf_file_path = generate_pdf_with_images("C:/Users/gayat/OneDrive/Desktop/comic/static/image")
    return send_file(pdf_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)