'''Program to analyse the sentiment of 2 choices and compare which has a more positive sentiment'''

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# uncomment the next 2 lines for the first run of the program
# import nltk
# nltk.download('all')

def preprocess_text(text):
    '''function for preprocessing text'''
  
    # tokenize
    tokens = word_tokenize(text.lower())

    # remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # join the tokens into a string
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

# initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    '''function to get the sentiment of a string'''
    
    scores = analyzer.polarity_scores(text)

    return scores['compound']

# get user input and place into a dictionary with key being title and value being description
choices = {}
while True:
    choice = input("Describe a choice (or type 'NIL' to finish): ")
    if choice == 'NIL':
        break
    description = input("Enter your thought about this choice (pros and cons): ")
    choices[choice] = description

# preprocess the text descriptions
for choice in choices:
    choices[choice] = preprocess_text(choices[choice])

# replace descriptions with sentiment scores
for choice in choices:
    choices[choice] = get_sentiment(choices[choice])

# sort the choices by sentiment score in descending order
choices = dict(sorted(choices.items(), key=lambda item: item[1], reverse=True))

# print the sentiment scores
print("Choices ranked by how positive your thoughts about them are from -1 to 1:")
for i, choice in enumerate(choices, 1):
    print(f"{i}. {choice}: {choices[choice]}")
