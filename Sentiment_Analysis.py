from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# uncomment the next 2 lines for the first run of the program
# import nltk
# nltk.download('all')

# get user input
one = input("Explain the first choice: ")
two = input("Explain the second choice: ")


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

# apply the function on the inputs
one_processed = preprocess_text(one)
two_processed = preprocess_text(two)

# initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    '''function to get the sentiment of a string'''
    
    scores = analyzer.polarity_scores(text)

    return scores['compound']

# apply get_sentiment function
one_sentiment = get_sentiment(one_processed)
two_sentiment = get_sentiment(two_processed)

print("First choice: " + str(one_sentiment))
print("Second choice: " + str(two_sentiment))
