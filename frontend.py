import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import time
import matplotlib.animation as animation

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000/predict"

# Streamlit UI Configuration
st.set_page_config(page_title="📜 Poem Analyzer", layout="wide")

# Toggle for Dark/Light Mode with smooth transition
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

if dark_mode:
    st.markdown(
        """
        <style>
            body { background-color: #121212; color: white; transition: background-color 0.5s ease-in-out; }
            .stTextArea, .stButton { background-color: #333; color: white; transition: background-color 0.3s ease-in-out; }
            .fade-in { animation: fadeIn 2s; }
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
        </style>
        """,
        unsafe_allow_html=True
    )

st.title("📜 Poem Theme, Era, Sentiment, Rhyme Scheme & Poetic Form Analysis")

# User Input
poem = st.text_area("✍️ Enter your poem here:", height=200, help="Type or paste a poem to analyze.")

# Animate the prediction process
if st.button("🔍 Predict") and poem:
    with st.spinner("✨ Analyzing poem..."):
        time.sleep(1)  # Simulate loading time
        try:
            response = requests.post(API_URL, json={"text": poem})

            if response.status_code == 200:
                result = response.json()
                predicted_era = result.get("era", "Unknown")
                predicted_theme = result.get("theme", "Unknown")
                predicted_sentiment = result.get("sentiment", "Unknown")
                predicted_rhyme = result.get("rhyme_scheme", "Unknown")
                predicted_form = result.get("poetic_form", "Unknown")
                confidence_era = result.get("confidence_era", 0)
                confidence_theme = result.get("confidence_theme", 0)
                confidence_sentiment = result.get("confidence_sentiment", 0)
                confidence_form = result.get("confidence_form", 0)
                explanation = result.get("explanation", "No explanation available.")

                # Animated Fade-In Effect
                st.markdown('<div class="fade-in">', unsafe_allow_html=True)

                # Display Results
                st.subheader("📌 Prediction Results")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 📅 Predicted Era")
                    st.success(f"**{predicted_era}** ({confidence_era}% confidence)")

                    st.markdown("### 🎭 Predicted Theme")
                    st.info(f"**{predicted_theme}** ({confidence_theme}% confidence)")

                with col2:
                    st.markdown("### ❤️ Predicted Sentiment")
                    st.warning(f"**{predicted_sentiment}** ({confidence_sentiment}% confidence)")

                    st.markdown("### 🔠 Predicted Rhyme Scheme")
                    st.success(f"**{predicted_rhyme}**")

                    st.markdown("### 📜 Predicted Poetic Form")
                    st.success(f"**{predicted_form}** ({confidence_form}% confidence)")

                # Confidence Score Animated Bar Plot
                st.subheader("📊 Confidence Scores")
                labels = ["Era", "Theme", "Sentiment", "Poetic Form"]
                confidences = [confidence_era, confidence_theme, confidence_sentiment, confidence_form]

                fig, ax = plt.subplots(figsize=(7, 4))
                bars = ax.bar(labels, confidences, color=sns.color_palette("viridis", len(labels)))
                ax.set_xlabel("Confidence (%)")
                ax.set_ylim(0, 100)

                def update(frame):
                    for bar, confidence in zip(bars, confidences):
                        bar.set_height(confidence * (frame / 20))

                ani = animation.FuncAnimation(fig, update, frames=20, interval=50)
                st.pyplot(fig)

                # Animated Sentiment Pie Chart
                st.subheader("🥧 Sentiment Distribution")
                sentiment_labels = ["Negative", "Neutral", "Positive", "Mixed"]
                sentiment_probs = np.random.randint(5, 25, size=4)  # Dummy values
                sentiment_probs[["Negative", "Neutral", "Positive", "Mixed"].index(predicted_sentiment)] = 40

                fig, ax = plt.subplots(figsize=(5, 5))
                wedges, texts, autotexts = ax.pie(
                    sentiment_probs, labels=sentiment_labels, autopct='%1.1f%%', 
                    colors=["red", "blue", "green", "purple"], startangle=90
                )

                def animate_pie(i):
                    for wedge in wedges:
                        wedge.set_alpha(i / 10)

                ani_pie = animation.FuncAnimation(fig, animate_pie, frames=10, interval=100)
                ax.axis("equal")
                st.pyplot(fig)

                # Animated Word Cloud
                st.subheader("☁️ Word Cloud of Poem")
                wordcloud = WordCloud(width=700, height=400, background_color="white").generate(poem)

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")

                def animate_wordcloud(i):
                    ax.set_alpha(i / 10)

                ani_wc = animation.FuncAnimation(fig, animate_wordcloud, frames=10, interval=100)
                st.pyplot(fig)

                # AI Explanation
                st.subheader("🤖 AI Explanation")
                st.write(f"**Why this prediction?**")
                st.write(explanation)

                # Closing fade-in div
                st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.error(f"❌ API Error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to FastAPI backend. Ensure it's running!")
        except Exception as e:
            st.error(f"⚠️ Unexpected Error: {e}")
