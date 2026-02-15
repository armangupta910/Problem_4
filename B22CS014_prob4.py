import time
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    """Fetches and prepares the Sports vs Politics dataset."""
    print("Downloading/Loading 20 Newsgroups dataset...")
    
    # Define the categories we want
    sports_cats = ['rec.sport.baseball', 'rec.sport.hockey']
    politics_cats = ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast']
    
    # Fetch data
    dataset = fetch_20newsgroups(subset='all', categories=sports_cats + politics_cats, 
                                 remove=('headers', 'footers', 'quotes'))
    
    # Create a DataFrame
    df = pd.DataFrame({'text': dataset.data, 'target': dataset.target})
    
    # Map targets to binary: 0 for Sports, 1 for Politics
    # In the fetched dataset, the indices map alphabetically to the categories.
    # Politics categories will have higher index numbers than rec.sport
    df['label'] = df['target'].apply(lambda x: 1 if dataset.target_names[x].startswith('talk.politics') else 0)
    
    # Drop empty documents
    df = df[df['text'].str.strip().astype(bool)]
    return df

def main():
    # 1. Load Data
    df = load_data()
    print(f"Total documents loaded: {len(df)}")
    print(f"Sports Docs: {len(df[df['label'] == 0])} | Politics Docs: {len(df[df['label'] == 1])}\n")
    
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    # 2. Feature Extraction: TF-IDF with N-grams (Unigrams + Bigrams)
    print("Extracting features using TF-IDF (Unigrams + Bigrams)...")
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 3. Define the three ML Models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine (Linear SVC)": LinearSVC(dual=False)
    }
    
    # 4. Train, Evaluate, and Compare
    results = []
    
    print("-" * 50)
    for name, model in models.items():
        start_time = time.time()
        
        # Train
        model.fit(X_train_vec, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = model.predict(X_test_vec)
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": acc, "Training Time (s)": train_time})
        
        print(f"Model: {name}")
        print(f"Accuracy: {acc * 100:.2f}%")
        print(classification_report(y_test, y_pred, target_names=["Sports", "Politics"]))
        print("-" * 50)
        
    # Print Summary Table
    results_df = pd.DataFrame(results)
    print("\nQuantitative Comparison Summary:")
    print(results_df.to_markdown(index=False))

if __name__ == "__main__":
    main()