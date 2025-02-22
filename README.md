# 📜 NLP Poem Analyzer  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)  
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-brightgreen)  
![License](https://img.shields.io/badge/License-MIT-yellow.svg)  
![Status](https://img.shields.io/badge/Status-Active-green)  

A powerful **NLP-based Poem Analyzer** that extracts **sentiment, themes, and stylistic elements** from poetry using Natural Language Processing techniques. 📖✨  

---

## 🚀 Features  

- ✅ **Sentiment Analysis** – Determines the poem’s emotional tone.  
- ✅ **Theme Extraction** – Identifies recurring motifs using topic modeling.  
- ✅ **Stylistic Analysis** – Detects metaphors, similes, and alliteration.  
- ✅ **Keyword Extraction** – Highlights important words and phrases.  
- ✅ **Visualization** – Generates charts to represent sentiment shifts.  

---

## ⚙️ How It Works  

1. **Preprocessing** – Cleans text, removes stop words, tokenizes.  
2. **Feature Extraction** – Uses **TF-IDF** and **word embeddings**.  
3. **Modeling** – Applies **sentiment analysis** & **topic modeling (LDA)**.  
4. **Visualization** – Displays **sentiment graphs & word clouds**.  

---

## 📥 Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/nlp-poem-analyzer.git
cd nlp-poem-analyzer
# 🎭 Poem Analyzer

A Python-based tool for analyzing poetry, extracting sentiment, themes, and literary devices, and generating visual insights.

---

## 2️⃣ Create a Virtual Environment

Create a virtual environment by running:

```bash
python -m venv venv
```

Activate the virtual environment:

For macOS/Linux:

```bash
source venv/bin/activate
```

For Windows (CMD/PowerShell):

```bash
venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

Once the virtual environment is activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the analyzer to process a poem:

```bash
python analyze.py --input "path/to/poem.txt"
```

For example:

```bash
python analyze.py --input "poems/sonnet.txt"
```

This command will output sentiment, themes, and stylistic insights, and generate visualizations saved in the `output/` folder.

---

## 📊 Example Output

```yaml
📜 Poem: Sonnet 18  
📈 Sentiment Score: Positive (0.87)  
🌿 Detected Themes: Love, Nature, Time  
✨ Key Literary Devices: Metaphor, Imagery, Personification  
📊 Word Cloud & Sentiment Graph: Saved in output/
```

---

## 🛠 Tech Stack

**Programming Language:** Python 🐍

**Libraries Used:**
- NLTK – Text processing
- spaCy – NLP tasks
- TextBlob – Sentiment analysis
- Matplotlib / Seaborn – Data visualization
- Scikit-learn – Machine learning

---

## 🤝 Contribution Guidelines

We welcome contributions! Follow these steps:

1. Fork this repository.
2. Create a new branch (e.g., `feature-branch`).
3. Commit your changes with meaningful messages.
4. Push to your fork and submit a Pull Request (PR).
5. 🎉 Your changes will be reviewed and merged!

For major changes, please open an issue first to discuss your ideas.

---

## 📬 Support

If you find this project useful, please consider:

- ⭐ Starring this repository
- 🐛 Reporting bugs via [Issues](https://github.com/RahulNair2004/repository/issues)
- 💡 Suggesting features via [Discussions](https://github.com/RahulNair2004/repository/discussions)

---

## 📝 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

Happy Coding! 🎭✨
