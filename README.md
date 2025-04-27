# 📚 BookFeels Emotion-Based Book Recommendation System
 
## ✨ Introduction
BookFeels is an emotion-driven book recommendation system that uses natural language processing and machine learning to suggest books based on the emotional tone extracted from their reviews.
Developed as a graduation project, it combines lexicon-based emotion detection, deep learning sentiment analysis, and a hybrid recommendation engine.

## 🚀 Features
- Emotion Detection from book reviews (happiness, sadness, excitement, etc.)

- Lexicon Expansion using WordNet to cover more emotional expressions

- Negation Handling (e.g., “not happy” ➔ interpreted as sad)

- Sentiment Analysis using pre-trained RoBERTa model (Twitter-roBERTa)

- A Hybrid Recommendation System combining emotions, book metadata, and ratings

- Modular Design for flexibility and customization

## 🗂 Dataset
- Book Reviews with titles, authors, categories, review texts, and ratings

- Preprocessing includes text cleaning, lemmatization, and negation handling

- Emotion Profiles are aggregated at the book level

## 🛠️ Approach
### Emotion Classification:
- Expand a base emotion lexicon using WordNet synonyms/antonyms, then detect emotions per review.

### Sentiment Analysis:
- Use CardiffNLP’s Twitter-RoBERTa to classify reviews as positive/neutral/negative.

### BERT Fine-Tuning (Experimental):
- Fine-tune BERT on Google’s GoEmotions dataset (27 emotion labels) to understand multi-label emotion classification.

## Recommendation Logic:

- TF-IDF vectorization of emotions, author, and genre

- Cosine similarity matching between user emotions and books

- Rating normalization with MinMaxScaler

- Final ranking combining similarity and rating

## 🛠 Tech Stack
- Languages: Python 3.x

- Libraries: pandas, numpy, nltk, scikit-learn, transformers, torch

- Tools: Jupyter Notebooks

- Models: RoBERTa (for sentiment), experimental BERT (for emotions)

# 📋 How to Run
Clone the repo:

bash
Copy
Edit
git clone https://github.com/ran-ibra/BookFeels.git
cd BookFeels
Install dependencies:

bash
Copy
Edit
pip install pandas numpy scipy nltk torch transformers scikit-learn
Download NLTK data:

python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
Prepare Dataset:
Place your reviews CSV in the working directory.

Run the notebooks:

emotion_classification.ipynb ➔ Classifies emotions in reviews

Recommendation_&_Book_emotion_classification.ipynb ➔ Builds recommendations

## 💬 Acknowledgements
- Hugging Face (Transformers Library)

- CardiffNLP RoBERTa Sentiment Model

- Google’s GoEmotions Dataset

- Open-source Python and NLP community ❤️

# 📝 Final Note
# This project represents a memorable and emotional journey — building something from scratch, overcoming hardware limitations, and combining my love for books and AI into one project.

