import os
import sys
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from phase2_Model.model import recommend_jobs, skill_list

app = Flask(__name__, template_folder="templates", static_folder="static")

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

# ----------------------------------------
@app.route("/recommend", methods=["POST"])
def recommend():

    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "message": "No data received."
        }), 400

    skills = data.get("skills", [])

    if len(skills) == 0:
        return jsonify({
            "success": False,
            "message": "Please enter at least one skill."
        }), 400

    try:
        recommendations = recommend_jobs(user_skills=skills, top_n=10, gap_top_n=5, n_similar=30)
        return jsonify({
            "success": True,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route("/skills")
def get_skills():
    return jsonify(skill_list)

# ----------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )