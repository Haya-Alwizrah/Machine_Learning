from flask import Flask, render_template, request, jsonify
from Project.phase2_Model.model import recommend_jobs

app = Flask(__name__)

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route("/recommend", methods=["POST"])

def recommend():

    data = request.get_json()

    skills = data["skills"]

    results = recommend_jobs(skills)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)