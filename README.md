# 💤 Sleep Disorder Detection and Analysis

This project is a machine learning and web-based solution designed to detect sleep disorders using lifestyle and health-related inputs.
It combines data analysis, predictive modeling, and a user-friendly Flask web interface to help users understand their sleep health. 
Users can fill out a survey on the website, and based on their responses (like age, gender, sleep duration, stress level, physical activity, etc.),
the model predicts if they are likely to have a sleep disorder. The web app also includes additional features like a sleep journal, facts about sleep health, a sleep tracker,
tips and recommendations for better sleep, calming music, downloadable reports, a help centre and a feedback form.

The machine learning models used include Logistic Regression (best accuracy model in my project), XGBoost, CatBoost, Gradient Boosting, and SVM. 
Each model was trained and evaluated using performance metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
These models were implemented in Jupyter Notebook, where the initial data analysis and preprocessing were also done.

In the Jupyter Notebook part of the project, detailed exploratory data analysis (EDA) was performed using visualizations such as box plots, bar graphs,3-D scatter plots,
heatmaps, pair plots, Tree Maps, and correlation matrices. This helped uncover patterns and insights, such as the impact of stress levels, occupation, and work hours on 
sleep disorders. The final trained model was saved and integrated into the Flask application to give real-time predictions.

Overall, this project aims to create awareness about sleep disorders using data science and make it accessible in a friendly, aesthetic web format.
