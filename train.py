"""
Train a sentiment classifier for Amazon reviews.

Dataset labels:
    __label__1 = negative review
    __label__2 = positive review

Install the required packages:
    pip install scikit-learn joblib

Example:
    python train.py --train-file "C:/Users/1/Downloads/archive/train.ft.txt/train.ft.txt" \
        --test-file "C:/Users/1/Downloads/archive/test.ft.txt/test.ft.txt"
"""

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report


def read_reviews(file_path, batch_size, max_samples=None):
    """Read reviews in batches so the whole dataset isn't loaded into memory."""
    review_texts = []
    sentiment_labels = []
    total_read = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            # Each line begins with a label followed by the review text.
            match = re.match(r"^(__label__[12])\s+(.*)$", line.rstrip())

            # Skip lines that don't match the expected dataset format.
            if not match:
                continue

            label, review = match.groups()

            # Convert the dataset labels to numbers:
            # 0 means negative, 1 means positive.
            sentiment_labels.append(0 if label == "__label__1" else 1)
            review_texts.append(review)
            total_read += 1

            # Yield a batch when it reaches the requested size.
            if len(review_texts) >= batch_size:
                yield review_texts, np.array(sentiment_labels)
                review_texts = []
                sentiment_labels = []

            # Stop early if a sample limit was specified.
            if max_samples is not None and total_read >= max_samples:
                break

    # Yield any reviews left over after the final full batch.
    if review_texts:
        yield review_texts, np.array(sentiment_labels)


def main():
    # Command-line options let you specify the dataset files and training settings.
    parser = argparse.ArgumentParser(description="Train an Amazon review sentiment model.")
    parser.add_argument("--train-file", required=True, help="Path to the training data file")
    parser.add_argument("--test-file", help="Optional path to the test data file")
    parser.add_argument("--model-out", default="sentiment_model.joblib", help="Where to save the model")
    parser.add_argument("--batch-size", type=int, default=4096, help="Reviews processed per batch")
    parser.add_argument("--max-train-samples", type=int, help="Optional training sample limit")
    parser.add_argument("--max-test-samples", type=int, help="Optional test sample limit")
    args = parser.parse_args()

    train_path = Path(args.train_file)
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    # Convert review text into numeric word features.
    # HashingVectorizer can process the large dataset without storing a vocabulary.
    vectorizer = HashingVectorizer(
        n_features=2**20,
        alternate_sign=False,
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
    )

    # Logistic regression is a classification method.
    # SGDClassifier trains it incrementally, batch by batch.
    model = SGDClassifier(
        loss="log_loss",
        random_state=42,
    )

    # Train the model using the labeled reviews.
    first_batch = True
    training_count = 0

    for reviews, labels in read_reviews(
        train_path,
        args.batch_size,
        args.max_train_samples,
    ):
        # Transform the text into numeric features for the model.
        review_features = vectorizer.transform(reviews)

        # The first batch tells the classifier which classes exist.
        if first_batch:
            model.partial_fit(review_features, labels, classes=np.array([0, 1]))
            first_batch = False
        else:
            model.partial_fit(review_features, labels)

        training_count += len(labels)
        print(f"Trained on {training_count:,} reviews")

    if first_batch:
        raise ValueError("No valid reviews found in the training file.")

    # Save the trained classifier and feature settings for the app to load.
    output_path = Path(args.model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "n_features": 2**20,
            "training_examples": training_count,
        },
        output_path,
    )
    print(f"Saved trained model to: {output_path.resolve()}")

    # Evaluate the model on test data if a test file was provided.
    if args.test_file:
        test_path = Path(args.test_file)
        if not test_path.is_file():
            raise FileNotFoundError(f"Test file not found: {test_path}")

        true_labels = []
        predicted_labels = []

        for reviews, labels in read_reviews(
            test_path,
            args.batch_size,
            args.max_test_samples,
        ):
            review_features = vectorizer.transform(reviews)
            predictions = model.predict(review_features)

            true_labels.extend(labels)
            predicted_labels.extend(predictions)

        if true_labels:
            print(f"Test accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")
            print(
                classification_report(
                    true_labels,
                    predicted_labels,
                    target_names=["Negative", "Positive"],
                )
            )


if __name__ == "__main__":
    main()