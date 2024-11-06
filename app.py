from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# โหลดข้อมูลและสร้างโมเดล
files = ['INFP_cleaned.csv', 'INFJ_cleaned.csv', 'INTJ_cleaned.csv', 'INTP_cleaned.csv']
dfs = [pd.read_csv(file) for file in files]
data = pd.concat(dfs, ignore_index=True)

mbti_to_animal = {'INTP': 'Owl', 'INTJ': 'Cat', 'INFP': 'Dolphin', 'INFJ': 'Wolf'}
data['animal'] = data['mbti_type'].map(mbti_to_animal)
data['comment'].fillna('', inplace=True)

# สร้าง pipeline สำหรับโมเดล
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['comment'], data['animal'], test_size=0.2, random_state=42)

# ฝึกโมเดล
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

pipeline.fit(X_train, y_train)

# ฟังก์ชันสำหรับทำนายประเภทสัตว์
def predict_animal(comment):
    animal = pipeline.predict([comment])[0]
    return animal

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_comment = request.form['comment']
        predicted_animal = predict_animal(user_comment)
        return render_template('index.html', animal=predicted_animal)

if __name__ == '__main__':
    app.run(debug=True)
