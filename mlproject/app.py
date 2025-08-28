from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import joblib
from konlpy.tag import Okt

app = Flask(__name__) # __name__은 파일 이름
CORS(app)

okt = Okt()

try:
    model = joblib.load("model/lr_v1.pkl")
    vec = joblib.load("model/tfidf_vec_v1.pkl")
except Exception as e:
    print('모델 로드 중 오류 발생: {str(e)}')
    raise

@app.route("/")
def hello_world():
    return "<p>안녕하세요! /ai로 가세요!</p>"

@app.route("/ai")
def hello_world2():
    return render_template("index.html") # templates 폴더를 만들고 index.html을 작성하세요.

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() # form으로 받은 걸 json으로 바꿔줘!
    if not data or "text" not in data:
        return jsonify({"error": "텍스트가 흠......"})
    text = data['text']
    if not text.strip():
        return jsonify({"error": "이건..... 텍스트가 흠......"})
    text_tfidf = vec.transform([text])
    predict = model.predict(text_tfidf)[0]
    return jsonify({"emotion": '긍정'}) # str(predict)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)