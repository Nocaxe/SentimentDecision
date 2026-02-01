'''Program to analyse the sentiment of multiple choices using ML-based sentiment analysis'''

import warnings
from transformers import pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_sentiment(text):
    '''Function to get the sentiment intensity score of a string using ML model'''

    # Get predictions from the model
    results = analyzer(text)[0]
    print(results)

    # Extract individual scores
    negative_score = next(item['score'] for item in results if item['label'] == 'negative')
    positive_score = next(item['score'] for item in results if item['label'] == 'positive')

    # Calculate compound score between -1 and 1
    compound_score = positive_score - negative_score

    return compound_score

# Initialize sentiment analyzer (uses RoBERTa model)
print("Loading sentiment analysis model...")
analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=-1,  # Use CPU; change to 0 for GPU if available
    top_k=None  # Get probabilities for all sentiment classes
)
print("Model loaded successfully!\n")

# get user input and place into a dictionary with key being title and value being description
choices = {}
while True:
    choice = input("\nDescribe a choice (or press ENTER to finish): ")
    if choice == '':
        break
    description = input("Enter your thought about this choice (pros and cons): ")
    choices[choice] = description

# calculate sentiment scores using ML model
print("\nAnalyzing sentiment...")
for choice in choices:
    choices[choice] = get_sentiment(choices[choice])

# sort the choices by sentiment score in descending order
choices = dict(sorted(choices.items(), key=lambda item: item[1], reverse=True))

# print the sentiment scores
print("Choices ranked by how positive your thoughts about them are from -1 to 1:")
for i, choice in enumerate(choices, 1):
    print(f"{i}. {choice}: {choices[choice]}")
