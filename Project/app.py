from flask import Flask, render_template, request, jsonify
from model import all_skills, get_recommendations

app = Flask(__name__)

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/try')
def try_model():
    return render_template('try.html', available_skills=all_skills)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_skills = data.get('skills', [])
    
    if not user_skills:
        return jsonify({"jobs": [], "missing_skills": []})
        
    results = get_recommendations(user_skills)
    
    return jsonify(results)

@app.route('/api/skills')
def get_skills():
    return jsonify(all_skills)

if __name__ == '__main__':
    app.run(debug=True)