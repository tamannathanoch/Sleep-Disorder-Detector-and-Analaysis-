from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
from datetime import datetime
import os
import re
import pickle
from xhtml2pdf import pisa
from io import BytesIO

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load trained model
MODEL_PATH = "sleep_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# File paths
dataset_path = "data/Sleep_health_and_lifestyle_dataset.csv"
user_data_path = "data/user_inputs.csv"

if not os.path.exists(user_data_path):
    original = pd.read_csv(dataset_path)
    pd.DataFrame(columns=original.columns).to_csv(user_data_path, index=False)

def load_users():
    if not os.path.exists("users.csv"):
        return {}
    df = pd.read_csv("users.csv")
    return dict(zip(df.email, df.password))

def save_user(email, password):
    if not os.path.exists("users.csv"):
        pd.DataFrame(columns=["email", "password"]).to_csv("users.csv", index=False)
    df = pd.read_csv("users.csv")
    df.loc[len(df.index)] = [email, password]
    df.to_csv("users.csv", index=False)

@app.route("/")
def home():
    return render_template("home.html", page="home")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if not re.match(r"^[a-zA-Z0-9._%+-]+@gmail\.com$", email):
            return render_template("signup.html", error="Invalid email format. Use @gmail.com.")

        users = load_users()
        if email in users:
            return render_template("signup.html", error="Email already exists.")

        save_user(email, password)
        session["logged_in"] = True
        session["email"] = email
        return redirect(url_for("form"))

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        users = load_users()
        if email in users and users[email] == password:
            session["logged_in"] = True
            session["email"] = email
            return redirect(url_for("form"))
        else:
            return render_template("login.html", error="Invalid credentials.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/form", methods=["GET", "POST"])
def form():
    if "logged_in" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        form = request.form

        gender_map = {"Male": 1, "Female": 0, "Other": 2}
        bmi_map = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
        occ_map = {
            "Student": 0, "Employed": 1, "Self-employed": 2,
            "Unemployed": 3, "Retired": 4, "Other": 5
        }

        def extract_systolic(bp):
            try:
                return int(bp.split('/')[0])
            except:
                return 120

        data = {
            "Gender": gender_map.get(form["gender"], 2),
            "Age": int(form["age"]),
            "Occupation": occ_map.get(form["occupation"].strip().title(), 5),
            "Sleep Duration": float(form["sleep_duration"]),
            "Quality of Sleep": int(form["quality_of_sleep"]),
            "Physical Activity Level": int(form["physical_activity"]),
            "Stress Level": int(form["stress_level"]),
            "BMI Category": bmi_map.get(form["bmi_category"], 1),
            "Blood Pressure": extract_systolic(form["blood_pressure"]),
            "Heart Rate": int(form["heart_rate"]),
            "Daily Steps": int(form["daily_steps"])
        }

        df = pd.DataFrame([data])[model.feature_names_in_]
        pred = model.predict(df)[0]

        label_map = {0: "Normal", 1: "Insomnia", 2: "Sleep Apnea"}
        pred_label = int(pred)
        pred_str = label_map.get(pred_label, "Unknown")

        # === Risk Calculation ===
        risk_score = 0

        stress = int(form["stress_level"])
        risk_score += (max(0, stress - 6) / 4) * 25

        sleep = float(form["sleep_duration"])
        if sleep < 6.5 or sleep > 8.5:
            risk_score += 20

        activity = int(form["physical_activity"])
        if activity < 5:
            risk_score += (5 - activity) * 3

        bp = extract_systolic(form["blood_pressure"])
        if bp < 100 or bp > 130:
            risk_score += 20

        hr = int(form["heart_rate"])
        if hr < 60 or hr > 100:
            risk_score += 20

        # === Risk to % Mapping with Label Override ===
        if risk_score <= 20:
            risk_percentage = 20
            sentiment = "positive"
            pred_str = "Normal"  # ⬅ override label
            tips = [
                "You're doing well! Maintain a consistent routine.",
                "Aim for 7–8 hours of sleep daily.",
                "Keep up with your physical activity and hydration."
            ]
        elif risk_score <= 50:
            risk_percentage = 50
            sentiment = "neutral"
            tips = [
                "Reduce screen time in the evenings.",
                "Avoid caffeine after 5 PM.",
                "Try meditation or journaling before bed."
            ]
        else:
            risk_percentage = 80
            sentiment = "negative"
            tips = [
                "High risk detected. Please consult a sleep expert.",
                "Stick to a strict sleep-wake routine.",
                "Avoid heavy meals and phone use before sleep."
            ]

        reverse_gender = {1: "Male", 0: "Female", 2: "Other"}
        reverse_bmi = {0: "Underweight", 1: "Normal", 2: "Overweight", 3: "Obese"}
        reverse_occ = {
            0: "Student", 1: "Employed", 2: "Self-employed",
            3: "Unemployed", 4: "Retired", 5: "Other"
        }

        readable_data = data.copy()
        readable_data["Gender"] = reverse_gender.get(data["Gender"])
        readable_data["BMI Category"] = reverse_bmi.get(data["BMI Category"])
        readable_data["Occupation"] = reverse_occ.get(data["Occupation"])

        clean_data = {k: int(v) if isinstance(v, (int, float)) else str(v) for k, v in readable_data.items()}

        session["result_data"] = {
            "prediction_result": pred_str,
            "risk_percentage": int(risk_percentage),
            "sentiment": sentiment,
            "user_data": clean_data,
            "health_tips": tips
        }

        session["sleep_quality"] = pred_str
        session["disorder_risk"] = pred_str
        session["sleep_duration"] = float(form["sleep_duration"])
        session["recommendation"] = tips[0]

        return redirect(url_for("result"))

    return render_template("form.html")

@app.route("/result")
def result():
    if "logged_in" not in session:
        return redirect(url_for("login"))

    result_data = session.get("result_data")
    if not result_data:
        return redirect(url_for("form"))

    return render_template(
        "result.html",
        prediction_result=result_data["prediction_result"],
        risk_percentage=result_data["risk_percentage"],
        sentiment=result_data["sentiment"],
        user_data=result_data["user_data"],
        health_tips=result_data["health_tips"]
    )

@app.route("/visual")
def visual():
    images = [img for img in os.listdir("static/visualizations") if img.endswith(".png")]
    html_visuals = [file for file in os.listdir("static/visualizations") if file.endswith(".html")]
    return render_template("visual.html", images=images, html_visuals=html_visuals)

@app.route("/tracker", methods=["GET", "POST"])
def tracker():
    tracker_data = []
    csv_path = os.path.join("data", "sleep_log.csv")

    if request.method == "POST":
        entry = {
            "date": request.form["date"],
            "sleep_time": request.form["sleep_time"],
            "wake_time": request.form["wake_time"],
            "sleep_quality": request.form["sleep_quality"],
            "mood": request.form["mood"],
            "notes": request.form["notes"]
        }

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame()

        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv(csv_path, index=False)

    if os.path.exists(csv_path):
        tracker_data = pd.read_csv(csv_path).to_dict(orient="records")

    return render_template("tracker.html", tracker_data=tracker_data, page="tracker")

@app.route("/recommendations")
def recommendations():
    return render_template("recommendations.html")

@app.route("/report")
def report():
    return render_template("reportdownload.html")

@app.route("/reportdownload", methods=["GET", "POST"])
def reportdownload():
    if "logged_in" not in session:
        return redirect(url_for("login"))

    result_data = session.get("result_data", {})
    report_data = {
        "report_date": datetime.now().strftime("%Y-%m-%d"),
        "sleep_quality": result_data.get("prediction_result", "Unknown"),
        "disorder_risk": result_data.get("prediction_result", "Unknown"),
        "sleep_duration": session.get("sleep_duration", "N/A"),
        "recommendation": result_data.get("health_tips", ["Maintain a regular sleep routine."])[0]
    }

    if request.method == "POST":
        rendered = render_template("report_template.html", **report_data)
        pdf = BytesIO()
        pisa.CreatePDF(rendered, dest=pdf)
        pdf.seek(0)
        return send_file(pdf, as_attachment=True, download_name="Sleep_Report.pdf")

    return render_template("reportdownload.html", **report_data)

@app.route("/aboutus")
def about():
    return render_template("aboutus.html")

@app.route("/help")
def help():
    return render_template("help.html")

if __name__ == "__main__":
    app.run(debug=True)