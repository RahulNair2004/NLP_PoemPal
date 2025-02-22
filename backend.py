from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, pipeline
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ Load trained models for Era, Theme, and Poetic Form classification (XGBoost)
try:
    with open("tfidf_vectorizer1.pkl", "rb") as f:
        vectorizer_era = pickle.load(f)
    with open("xgb1.pkl", "rb") as f:
        model_era = pickle.load(f)
    with open("label_encoder1.pkl", "rb") as f:
        label_encoder_era = pickle.load(f)

    with open("theme_vectorizer.pkl", "rb") as f:
        vectorizer_theme = pickle.load(f)
    with open("theme_xgboost_model.pkl", "rb") as f:
        model_theme = pickle.load(f)
    with open("theme_label_encoder.pkl", "rb") as f:
        label_encoder_theme = pickle.load(f)

    with open("vectorizer_poetic_form.pkl", "rb") as f:
        vectorizer_form = pickle.load(f)
    with open("xgboost_poetic_form.pkl", "rb") as f:
        model_form = pickle.load(f)
    with open("label_encoder_poetic_form.pkl", "rb") as f:
        label_encoder_form = pickle.load(f)

except FileNotFoundError:
    raise FileNotFoundError("❌ Required vectorizers or models are missing! Ensure they are properly saved.")
except Exception as e:
    raise RuntimeError(f"⚠️ Error loading XGBoost models: {str(e)}")

# ✅ Load fine-tuned BERT model & tokenizer for Sentiment Analysis
try:
    model_name = "poem_sentiment_bert_model"
    sentiment_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    raise RuntimeError(f"⚠️ Error loading BERT Sentiment Model: {str(e)}")

# ✅ Load AI explanation model
try:
    explanation_model = pipeline("text-generation", model="facebook/opt-1.3b")
except Exception as e:
    explanation_model = None
    print(f"⚠️ AI explanation model failed to load: {e}")

# ✅ Load LSTM model for Rhyme Scheme
try:
    lstm_model = load_model("lstm_rhyme_model.keras")  # Keras 3 format
except Exception as e:
    raise RuntimeError(f"⚠️ Error loading LSTM model: {str(e)}")

# ✅ Load Tokenizer for LSTM
try:
    with open("tokenizer_rhyme.pkl", "rb") as f:
        tokenizer_rhyme = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError("❌ tokenizer_rhyme.pkl file is missing!")

# ✅ Load Label Encoder for Rhyme Schemes
try:
    with open("label_encoder_rhyme.pkl", "rb") as f:
        label_encoder_rhyme = pickle.load(f)
    if not hasattr(label_encoder_rhyme, "classes_"):
        raise ValueError("🚨 Label Encoder is not fitted properly!")
except FileNotFoundError:
    raise FileNotFoundError("❌ label_encoder_rhyme.pkl file is missing!")

# ✅ FastAPI app
app = FastAPI()

# ✅ Define input structure
class PoemInput(BaseModel):
    text: str

@app.post("/predict")
def predict_poem(poem: PoemInput):
    input_text = poem.text

    # ---- ERA PREDICTION (XGBoost) ----
    transformed_text_era = vectorizer_era.transform([input_text])
    predicted_label_era = model_era.predict(transformed_text_era)[0]
    predicted_era = label_encoder_era.inverse_transform([predicted_label_era])[0]
    confidence_era = round(np.max(model_era.predict_proba(transformed_text_era)) * 100, 2)

    # ---- THEME PREDICTION (XGBoost) ----
    transformed_text_theme = vectorizer_theme.transform([input_text])
    predicted_label_theme = model_theme.predict(transformed_text_theme)[0]
    predicted_theme = label_encoder_theme.inverse_transform([predicted_label_theme])[0]
    confidence_theme = round(np.max(model_theme.predict_proba(transformed_text_theme)) * 100, 2)

    # ---- SENTIMENT ANALYSIS (BERT) ----
    inputs = tokenizer(input_text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    logits = sentiment_model(**inputs).logits
    probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
    predicted_label_sentiment = np.argmax(probabilities)
    sentiment_label_map = {0: "Negative", 1: "Neutral", 2: "Positive", 3: "Mixed"}
    predicted_sentiment = sentiment_label_map[predicted_label_sentiment]
    confidence_sentiment = round(probabilities[predicted_label_sentiment] * 100, 2)

    # ---- RHYME SCHEME PREDICTION (LSTM) ----
    sequence = tokenizer_rhyme.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequence, maxlen=10)
    rhyme_prediction = lstm_model.predict(padded_sequence)
    predicted_rhyme_index = np.argmax(rhyme_prediction)
    predicted_rhyme = label_encoder_rhyme.inverse_transform([predicted_rhyme_index])[0]
    confidence_rhyme = round(np.max(rhyme_prediction) * 100, 2)

    # ---- POETIC FORM PREDICTION (XGBoost) ----
    transformed_text_form = vectorizer_form.transform([input_text]).toarray()  # Convert sparse matrix to dense array
    predicted_label_form = model_form.predict(transformed_text_form)[0]
    predicted_form = label_encoder_form.inverse_transform([predicted_label_form])[0]
    confidence_form = round(np.max(model_form.predict_proba(transformed_text_form)) * 100, 2)

    # ✅ AI-Generated Explanation
    explanation_prompt = (
        f"Analyze this poem: '{input_text}'. "
        f"Why is its theme classified as '{predicted_theme}', "
        f"its era as '{predicted_era}', "
        f"and its poetic form as '{predicted_form}'?"
    )

    if explanation_model:
        explanation = explanation_model(explanation_prompt, max_length=300, do_sample=True)[0]["generated_text"]
    else:
        explanation = "AI Explanation model is unavailable. Please try again later."

    return {
        "era": predicted_era,
        "theme": predicted_theme,
        "sentiment": predicted_sentiment,
        "rhyme_scheme": predicted_rhyme,
        "poetic_form": predicted_form,
        "confidence_era": confidence_era,
        "confidence_theme": confidence_theme,
        "confidence_sentiment": confidence_sentiment,
        "confidence_rhyme": confidence_rhyme,
        "confidence_form": confidence_form,
        "explanation": explanation
    }
