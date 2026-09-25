"""
Week 6: Encoding Text Data
Book Chapter: 6 - Encoding text data

Topics:
- CountVectorizer for n-gram frequency extraction
- TfidfVectorizer for term frequency-inverse document frequency encoding
- Combining text vectorizers with numeric/categorical preprocessors in ColumnTransformer
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def main():
    print("=" * 60)
    print(" Week 6: Encoding Text Data ")
    print("=" * 60)

    # Sample dataset containing text and numeric features
    df = pd.DataFrame({
        'text': [
            'Scikit-learn makes machine learning in Python easy and efficient.',
            'Deep learning requires GPU acceleration and neural networks.',
            'Machine learning pipelines prevent data leakage.',
            'Natural language processing involves text tokenization and vectorization.'
        ],
        'word_count': [9, 8, 6, 7],
        'topic': ['ML', 'DL', 'ML', 'NLP']
    })

    print("\n1. Text Vectorization with TfidfVectorizer:")
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    text_matrix = tfidf.fit_transform(df['text'])
    print(f"Extracted Vocabulary ({len(tfidf.get_feature_names_out())} features):")
    print(tfidf.get_feature_names_out())

    # 2. Multi-modal ColumnTransformer combining Text + Numeric features
    preprocessor = ColumnTransformer([
        ('text', TfidfVectorizer(stop_words='english'), 'text'),
        ('num', StandardScaler(), ['word_count'])
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])

    pipeline.fit(df[['text', 'word_count']], df['topic'])
    print("\n2. Combined Text + Numeric Pipeline fitted successfully!")

    sample_pred = pipeline.predict(pd.DataFrame({
        'text': ['Machine learning models require clean text preprocessing.'],
        'word_count': [7]
    }))
    print(f"Predicted topic for sample text: {sample_pred[0]}")

if __name__ == "__main__":
    main()
