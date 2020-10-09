import random
import string  # to process standard python strings
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True)  # for downloading packages


if input("Download Data?").lower().startswith('y'):
    nltk.download('punkt')  # first-time use only
    nltk.download('wordnet')  # first-time use only


# Reading in the corpus
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read()

# TOkenisation
sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(
        text.lower().translate(remove_punct_dict))
    )


# Keyword Matching
GREETING_INPUTS = [
    "hello", "hi", "greetings",
    "sup", "what's up", "hey"
]
GREETING_RESPONSES = [
    "hi", "hey", "*nods*", "hi there", "hello",
    "I am glad! You are talking to me"
]


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    _sent_tokens = sent_tokens.copy()
    _sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(_sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf < 0.5:
        return None
    else:
        return _sent_tokens[idx]


if __name__ == '__main__':
    flag = True
    print(
        "ROBO: My name is Robo. I will answer your queries about Chatbots. "
        "If you want to exit, type Bye!"
    )
    while flag is True:
        user_response = input()
        user_response = user_response.lower()
        if (user_response != 'bye'):
            if (user_response == 'thanks' or user_response == 'thank you'):
                flag = False
                print("ROBO: You are welcome..")
            else:
                if greeting(user_response) is not None:
                    print("ROBO: "+greeting(user_response))
                else:
                    print("ROBO: ", end="")
                    print(response(user_response))
        else:
            flag = False
            print("ROBO: Bye! take care..")
