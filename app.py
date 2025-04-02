from flask import Flask, render_template, redirect, url_for, request, session, jsonify
import random
from openai import OpenAI
import time
from datetime import datetime
import logging

# Set up logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

print("Server started at: ", datetime.now())
 
# Databricks LLM Setup
DATABRICKS_TOKEN = "dapi660f87007e8275d4f93556c4c60c4961"  # Replace with actual token
DATABRICKS_BASE_URL = "https://adb-911731324340270.10.azuredatabricks.net/serving-endpoints"
MODEL_NAME = "databricks-meta-llama-3-3-70b-instruct"
 
client = OpenAI(api_key=DATABRICKS_TOKEN, base_url=DATABRICKS_BASE_URL)
 
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management
 
leaderboard = [] # in-memory leaderboard storage

# Game difficulty levels
DIFFICULTY_LEVELS = {
    "level1": {
        "name": "I'm Just Starting to Learn About Medicine",
        "description": "Perfect for beginners! Learn the basics of the human body, basic health principles, and simple diagnosis techniques. A great starting point for anyone curious about medicine or health science.",
        "complexity": 0.5,
        "points_multiplier": 1
    },
    "level2": {
       "name": "I Have Some Knowledge, But I Want to Learn More",
        "description": "For those with some medical knowledge, this is your chance to dive deeper into more complex cases. Build on what you've learned and face situations similar to what you might encounter in a clinical setting. Ideal for aspiring doctors or health professionals!",
        "complexity": 0.7,
        "points_multiplier": 1.5
    },
    "level3": {
        "name": "I'm Ready to Take on Tough Medical Challenges",
        "description": "Ready for a challenge? This is for those who want to push their skills to the limit. Take on tough, life-or-death cases, and apply critical thinking in high-pressure situations. A great fit for pre-med students or anyone aiming for a career in healthcare!",
        "complexity": 0.9,
        "points_multiplier": 2
    }
}

# Function to get chat response from medical assistant
def get_medical_chat_response(message, chat_history=None):
    if chat_history is None:
        chat_history = []
    
    system_message = """You are DocMedi, a helpful and knowledgeable medical assistant chatbot.
    Your role is to:
    1. Provide clear, accurate health information in simple language
    2. Explain medical concepts without unnecessary jargon
    3. Acknowledge limitations of your knowledge when appropriate
    4. Maybe don't provide specific diagnoses - always recommend consulting healthcare professionals
    5. Be supportive and professional in your responses
    6. Keep responses concise and relevant to the user's questions

    """
    
    messages = [{"role": "system", "content": system_message}]
    
    # Add chat history to messages
    for entry in chat_history:
        messages.append({"role": "user" if entry["is_user"] else "assistant", "content": entry["message"]})
    
    # Add the current message
    messages.append({"role": "user", "content": message})
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=400,
            temperature=0.7,
            timeout=15  # Add explicit timeout
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Detailed LLM error: {type(e).__name__}: {str(e)}")
        raise  # Re-raise for the route handler to catch

def get_patient_complaint(level="level1"):
    complexity = DIFFICULTY_LEVELS[level]["complexity"]
    
    level_instructions = {
        "level1": "Use simple, everyday language. Focus on common, easily recognizable symptoms.",
        "level2": "Use moderately complex medical terms. Include some less obvious symptoms.",
        "level3": "Use advanced medical terminology. Describe complex symptom patterns."
    }
    
    syes_instr = f"""You are a patient visiting a doctor. {level_instructions[level]}  
                    Describe your symptoms in a familiar way, ensuring that each response corresponds to a unique disease Be restricted to max 100 words.  
                    """
    
    prompt = f"Generate a realistic patient complaint for a {DIFFICULTY_LEVELS[level]['name']} scenario. Complexity: {complexity*10}/10"
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": syes_instr},
                  {"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=complexity
    )
    return response.choices[0].message.content

def predict_disease(complaint, level="level1"):
    complexity = DIFFICULTY_LEVELS[level]["complexity"]
    
    prompt = f"Based on this complaint, predict the disease:\n\n{complaint}\n\nReturn only the disease name, using three words or less."
    
    sys_instr = f"""You are a medical expert. Predict the disease based on the given complaint.
                    Your response must be three words or less. Complexity level: {complexity*10}/10"""
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": sys_instr},
                  {"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=complexity/2
    )
    return response.choices[0].message.content.strip()

def get_disease_info(disease, level="level1"):
    complexity = DIFFICULTY_LEVELS[level]["complexity"]
    
    level_instructions = {
        "level1": "Use simple terms and basic explanations suitable for beginners.",
        "level2": "Use moderately complex terms and explanations suitable for medical students.",
        "level3": "Use advanced medical terminology and detailed explanations for medical professionals."
    }
    
    prompt = f"Provide a brief summary of the disease {disease}, including only two key bullet points for symptoms, causes, or precautions. Format the response in HTML."
    
    sys_instr = f'''You are a medical expert. Your response should be well-structured using HTML.
                   - The disease name should be in `<h2>` and centered.
                   - The summary should have only two bullet points and be left-aligned.
                   - Use `<ul>` and `<li>` for bullet points.
                   - The text should be short and concise. {level_instructions[level]}'''
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": sys_instr},
                  {"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=complexity/2
    )
    return response.choices[0].message.content.strip()

def generate_disease_options(correct_disease, level="level1"):
    disease_options = {
        "level1": [
            "Common Cold", "Flu", "Fever", "Food Poisoning", "Migraine", "Asthma", "Allergies", "Pneumonia",
            "Stomach Flu", "Ear Infection", "Pink Eye", "UTI", "Athlete's Foot", "Heartburn", "Dehydration",
            "Sunburn", "Poison Ivy Rash", "Sprained Ankle", "Insomnia", "Hangover"  
        ],
        "level2": [
            "Bronchitis", "Diabetes", "Hypertension", "Gastritis", "Arthritis", "Sinusitis", "Anemia",
            "High Cholesterol", "GERD", "Psoriasis", "Gout", "Hypothyroidism", "Hyperthyroidism", "Diverticulitis",
            "Kidney Stones", "Tendonitis", "Carpal Tunnel", "Chronic Fatigue", "IBS", "Sleep Apnea"
        ],
        "level3": [
            "Lupus", "Multiple Sclerosis", "Crohn's Disease", "Addison's Disease", "Kawasaki Disease",
            "Fibromyalgia", "Parkinson's", "ALS", "Cystic Fibrosis", "Sarcoidosis", "Ehlers-Danlos Syndrome",
            "Myasthenia Gravis", "Huntington's Disease", "Scleroderma", "Tourette Syndrome", "POTS",
            "Marfan Syndrome", "Wilson's Disease", "Guillain-Barré Syndrome", "CRPS (The 'Suicide Disease')"
        ]
    }
    
    all_diseases = disease_options[level]
    
    if correct_disease not in all_diseases:
        all_diseases.append(correct_disease)
    
    other_diseases = [d for d in all_diseases if d != correct_disease]
    options = [correct_disease] + random.sample(other_diseases, min(3, len(other_diseases)))
    random.shuffle(options)
    return options

@app.route('/')
def start():
    return render_template('start.html', leaderboard=leaderboard)

@app.route('/login', methods=['POST'])
def login():
    # Get user info from form
    name = request.form.get('name', 'Anonymous')
    email = request.form.get('email', '')
    age = request.form.get('age', '0')
    level = request.form.get('level', 'level1')
    
    # Store in session
    session['user_name'] = name
    session['user_email'] = email
    session['user_age'] = age
    session['game_level'] = level
    
    # Initialize game variables
    session['score'] = 0
    session['patients_left'] = 10
    
    return redirect(url_for('index'))

@app.route('/chat')
def chat():
    # Initialize chat history if it doesn't exist
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    user_name = session.get('user_name', 'Guest')
    
    # Add welcome message if this is a new chat
    if not session['chat_history']:
        welcome_message = f"Hello {user_name}! I'm DocMedi, your friendly medical assistant. How can I help you today? Feel free to ask me any medical questions, and I'll do my best to assist you."
        session['chat_history'] = [{"message": welcome_message, "is_user": False}]
        session.modified = True
    
    return render_template('chat.html', 
                          chat_history=session['chat_history'],
                          user_name=user_name)


@app.route('/instructions')
def instructions():
    return render_template('instructions.html')


@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        message = request.form.get('message', '')
        
        if not message.strip():
            return redirect(url_for('chat'))
        
        # Initialize chat history if needed
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        # Add user message to history
        session['chat_history'].append({"message": message, "is_user": True})
        
        # Simplified response handling with basic error management
        try:
            # Set a simple backup response in case of failure
            backup_response = "I'm having trouble accessing my medical knowledge right now. Could you try asking something else?"
            
            # Attempt to get LLM response with limited chat history (just the current message)
            # This avoids potential issues with large session data
            sys_instr = '''You are a medical expert. Format your response using:
            - Headings (`<h2>`)
            - Bullet points (`<ul>` and `<li>`)
            - Paragraphs (`<p>`)
            - Ensure only the title is centered (`<h2 style='text-align:center;'>`)
            - Provide a short summary with two  bullet points'''
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_instr},
                    {"role": "user", "content": message}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            chat_response = response.choices[0].message.content
            session['chat_history'].append({"message": chat_response, "is_user": False})
            
        except Exception:
            # Use backup response instead of trying detailed error handling
            session['chat_history'].append({"message": backup_response, "is_user": False})
        
        # Ensure session is saved
        session.modified = True
        return redirect(url_for('chat'))
        
    except Exception:
        # Last resort fallback
        return redirect(url_for('chat'))

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    # Clear chat history
    session['chat_history'] = []
    return redirect(url_for('chat'))

@app.route('/index')
def index():
    # If user not logged in, redirect to start page
    if 'user_name' not in session:
        return redirect(url_for('start'))
    
    level = session.get('game_level', 'level1')
    
    if 'score' not in session:
        session['score'] = 0
    if 'patients_left' not in session:
        session['patients_left'] = 10
    if 'current_patient' not in session or 'correct_disease' not in session:
        session['current_patient'] = get_patient_complaint(level)
        session['correct_disease'] = predict_disease(session['current_patient'], level)
        session['disease_options'] = generate_disease_options(session['correct_disease'], level)
    
    if session['patients_left'] < 0:
        session['patients_left'] = 10

    return render_template('index.html', 
                           current_patient=session['current_patient'],
                           disease_options=session['disease_options'],
                           score=session['score'], 
                           patients_left=session['patients_left'],
                           level=DIFFICULTY_LEVELS[level],
                           user_name=session['user_name'])
 
@app.route('/submit', methods=['POST'])
def submit():
    selected_disease = request.form['selected_disease']
    correct_disease = session['correct_disease']
    level = session.get('game_level', 'level1')
    points_multiplier = DIFFICULTY_LEVELS[level]['points_multiplier']
    
    if selected_disease == correct_disease:
        session['score'] += 10
        feedback = f"✅ Correct! You earned 10 coins."
        ans_zone="true";
    else:
        feedback = f"❌ Incorrect. The correct disease was {correct_disease} ."
        ans_zone="false";

    disease_info = get_disease_info(correct_disease, level)
    session['patients_left'] -= 1

    return render_template('index.html', 
                           ans_zone=ans_zone,
                           feedback=feedback, 
                           disease_info=disease_info,
                           current_patient=session['current_patient'], 
                           disease_options=session['disease_options'],
                           score=session['score'], 
                           patients_left=session['patients_left'],
                           level=DIFFICULTY_LEVELS[level],
                           user_name=session['user_name'])
 
@app.route('/next_patient', methods=['POST'])
def next_patient():
    level = session.get('game_level', 'level1')
    session['current_patient'] = get_patient_complaint(level)
    session['correct_disease'] = predict_disease(session['current_patient'], level)
    session['disease_options'] = generate_disease_options(session['correct_disease'], level)
    
    if session['patients_left'] == 0:
        return redirect(url_for('game_over'))
    
    return redirect(url_for('index'))
 
@app.route('/game_over', methods=['GET'])
def game_over():
    return render_template('game_over.html', 
                          score=session['score'], 
                          user_name=session.get('user_name', 'Anonymous'),
                          level_name=DIFFICULTY_LEVELS[session.get('game_level', 'level1')]['name'])
 
@app.route('/submit_score', methods=['POST'])
def submit_score():
    player_name = session.get('user_name', 'Anonymous')
    score = session.get('score', 0)
    level = session.get('game_level', 'level1')
    level_name = DIFFICULTY_LEVELS[level]['name']
    date = datetime.now().strftime("%Y-%m-%d")
    
    # Add to leaderboard
    leaderboard.append({
        'name': player_name,
        'score': score,
        'level': level_name,
        'date': date
    })
    
    # Sort leaderboard by score (highest first)
    leaderboard.sort(key=lambda x: x['score'], reverse=True)
    
    # Keep only top 10 scores
    if len(leaderboard) > 10:
        leaderboard[:] = leaderboard[:10]
        
    return redirect(url_for('start'))
 
@app.route('/restart_game', methods=['POST'])
def restart_game():
    # Clear only game-related session data, keep user info
    user_name = session.get('user_name')
    user_email = session.get('user_email')
    user_age = session.get('user_age')
    
    session.clear()
    
    # Restore user info
    if user_name:
        session['user_name'] = user_name
    if user_email:
        session['user_email'] = user_email
    if user_age:
        session['user_age'] = user_age
        
    return redirect(url_for('start'))

@app.errorhandler(500)
def handle_server_error(e):
    app.logger.error(f"500 error: {str(e)}", exc_info=True)
    return "Server error", 500
 
if __name__ == '__main__':
    app.run(debug=True)