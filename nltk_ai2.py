# Program to measure the similarity between
# two sentences using cosine similarity.
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentences import triggers, responses
from nltk.corpus import wordnet
from random import randint
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# X = input("Enter first string: ").lower()
# Y = input("Enter second string: ").lower()


def simplify(sentence):
    #tokenized = tokenizer.tokenize(sentence)
    words = word_tokenize(sentence)
    tagged = nltk.pos_tag(words)
    output = ''

    for i in range(0, len(words)):
        replacements = []

        # Only replace nouns with nouns, vowels with vowels etc.
        for syn in wordnet.synsets(words[i]):

            # Do not attempt to replace proper nouns or determiners
            if tagged[i][1] == 'NNP' or tagged[i][1] == 'DT':
                break
            
            # The tokenizer returns strings like NNP, VBP etc
            # but the wordnet synonyms has tags like .n.
            # So we extract the first character from NNP ie n
            # then we check if the dictionary word has a .n. or not
            word_type = tagged[i][1][0].lower()
            if syn.name().find("."+word_type+"."):
                # extract the word only
                r = syn.name()[0:syn.name().find(".")]
                replacements.append(r)

        if len(replacements) > 0:
            # Choose a random replacement
            replacement = replacements[randint(0, len(replacements)-1)]
            output = output + " " + replacement
        else:
            # If no replacement could be found, then just use the
            # original word
            output = output + " " + words[i]

    return output if output is not None else None


def filter_punct(sentence: str):
    sentence = sentence.casefold()
    punc = '''!()-[]{};:'"\\,<>./?@#$%^&*_~’'''

    for char in sentence:
        if char in punc:
            sentence = sentence.replace(char, "")
    return sentence


def confidence(_X, _Y):
    X = filter_punct(_X)
    Y = filter_punct(_Y)
    # tokenization
    X_list = word_tokenize(X)
    Y_list = word_tokenize(Y)

    # sw contains the list of stopwords
    sw = stopwords.words('english')
    l1 = []
    l2 = []

    # remove stop words from the string
    X_set = {w for w in X_list if w not in sw}
    Y_set = {w for w in Y_list if w not in sw}

    # form a set containing keywords of both strings
    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set:
            l1.append(1)  # create a vector
        else:
            l1.append(0)
        if w in Y_set:
            l2.append(1)
        else:
            l2.append(0)
    c = 0

    # cosine formula
    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    return cosine


def get_response(query):
    last_best = -1
    last_index = -1
    for trig in triggers:
        conf = confidence(query, trig)
        if last_best < conf:
            last_best = conf
            last_index = triggers[trig]

    return responses[last_index], last_best


if __name__ == '__main__':
    print("ROBO: How can I help?")
    while True:
        query = input("> ")
        response, conf = get_response(query)
        if conf < 0.8:
            print("ROBO: Sorry, I'm not sure.")
        else:
            print("ROBO: ", response)
            print("Confidence: ", conf)
