import joblib
# ሞዴሉን ካሰለጠንክ በኋላ ሴቭ ታደርገዋለህ
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
body {
  background-color: #f0f0f0; /* Set your desired background color */
}

button.back {
  background-color: #cccccc; /* Normal state */
}

button.back.active {
  background-color: #007bff; /* Active state */
  color: #fff;
}