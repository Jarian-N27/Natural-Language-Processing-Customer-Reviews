# Natural-Language-Processing-Customer-Reviews

🚧 **This README is a work in progress. Updates will be made soon.** 🚧

## 📌 Project Summary
This project focuses on developing a Natural Language Processing (NLP) model to automate sentiment analysis of customer feedback for a retail company. The objective is to compare traditional machine learning (ML) approaches (e.g., Naive Bayes, SVM, Random Forest) with a deep learning-based approach using a Transformer model from Hugging Face (DistilBERT) to classify reviews as positive, neutral, or negative.

## 🔧 Technologies & Tools
- **Programming Language:** Python
- **NLP Libraries:** [e.g., Hugging Face Transformers, spaCy, NLTK, Gensim]
- **Machine Learning Frameworks:** [e.g., TensorFlow, PyTorch, Scikit-learn]
- **Data Processing:** Pandas, NumPy
- **Vectorization & Embeddings:** [e.g., Word2Vec, BERT, OpenAI API]
- **Deployment:** [e.g., Flask, FastAPI, Streamlit, Docker]


### 🚀 Project Goals
- Develop a sentiment classification system that categorizes customer reviews into **positive, neutral, or negative**.
- Compare the effectiveness of **traditional ML algorithms** vs. **Deep Learning (Transformer-based)** approaches.
- Utilize **transfer learning** with **Hugging Face Transformers (DistilBERT)** for sentiment classification.
- Deploy the best-performing model for easy interaction and evaluation.

---

## 📊 Traditional Machine Learning Model Approach

### 🛠 Exploratory Data Analysis (EDA)
- Conducted an analysis of review distributions and key insights to understand sentiment trends.
- Identified potential biases, missing values, and class imbalances in the dataset.

### 🔍 Data Cleaning and Preprocessing
- Removed unnecessary characters, punctuation, and stopwords using **re** and **nltk**.
- Tokenized text and applied lemmatization.
- Vectorize data using TF-IDF vectorizer from Sci-kit Learn library.

### ⚙️ Model Selection & Training
- Implemented and compared traditional ML models:
  - **Naïve Bayes** (MultinomialNB)
  - **Random Forest Classifier**
  - **Logistic Regression**
  - **Neural Network- MLP(Multi Layer Perception) Classifier**
- Evaluated models using **accuracy, precision, recall, and F1-score**.

### 📈 Model Evaluation
- Performed cross-validation to ensure robustness.
- Plotted confusion matrices to analyze misclassifications.
- Identified the best-performing ML model for benchmarking against deep learning models.

---

## 🤖 Transformer Approach | Hugging Face Transformers

### 🔄 Data Preprocessing
- Utilized **DistilBERTTokenizer** for optimized tokenization.
- Converted text data into numerical representations suitable for transformer models.
- Ensured proper input sequence length and batch processing for efficient training.

### 🏗 Model Building
- Selected **DistilBERT**, a lightweight Transformer model, due to its balance between performance and efficiency.
- Fine-tuned the pre-trained model on our sentiment classification dataset.
- Implemented training with **PyTorch**.

### 📊 Model Evaluation
- Compared deep learning model performance against traditional ML models.
- Evaluated key metrics: **accuracy, precision, recall, F1-score, and ROC-AUC**.
- Fine-tuned hyperparameters to optimize model performance.

### 🌐 Model Deployment
- Deployed the trained Transformer model using **Streamlit** for an interactive web-based user interface.
- Enabled real-time sentiment classification of user input reviews.


## 🛠️ Installation & Usage
```bash
# Clone the repository
git clone https://github.com/yourusername/nlp_project.git
cd nlp_project

# Install dependencies
pip install -r requirements.txt

# Run the main script
python src/main.py
```

---

## 📌 Future Improvements
- Implement additional transformer architectures (e.g., **BERT, RoBERTa**) for comparison.
- Optimize hyperparameter tuning for both traditional ML and deep learning models.
- Integrate **explainability techniques** to interpret model predictions.
- Enhance the UI/UX of the Streamlit app with additional insights and analytics.

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.

## 📝 Acknowledgments
Special thanks to [Javier A. Dastas](https://github.com/javierdastas).

Link to the entire project [here](https://github.com/javierdastas/project-nlp-automated-customer-reviews)

