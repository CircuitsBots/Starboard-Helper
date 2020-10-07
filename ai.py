import pandas as pd
from ast import literal_eval

from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import download_model, download_bnpp_data
from cdqa.pipeline.cdqa_sklearn import QAPipeline

# Download data and models
#download_bnpp_data(dir='./data/bnpp_newsroom_v1.1/')

if input('Download model? Only do if you haven\'t already.').lower().startswith('y'):
    download_model(model='bert-squad_1.1', dir='./models')

df = pd.read_csv('data/data.csv', converters={'paragraphs': literal_eval})
#df = filter_paragraphs(df)

#df = pd.DataFrame(columns=['title', 'paragraphs'])
#paragraphs = input("Text to Analyze:\n").split('\n')
#df = df.append({'title': 'Inputed Data', 'paragraphs': paragraphs}, ignore_index=True)

#print(df)

cdqa_pipeline = QAPipeline(reader='models/bert_qa.joblib', min_df=1, max_df=1000)

cdqa_pipeline.fit_retriever(df=df)


class Response:
    def __init__(self, query, answer, title, paragraph, confidence):
        self.query = query
        self.answer = answer
        self.title = title
        self.paragraph = paragraph
        self.confidence = confidence

    def __str__(self):
        string = f"__**{self.title}**__\n"\
            f"{self.paragraph}\n"
        return string


class SimpleAnswer:
    def __init__(self, answer):
        self.answer = answer
        self.confidence = 10

    def __str__(self):
        return self.answer


def get_simple_anser(query):
    query = query.lower()

    def check(string):
        if query.startswith(string):
            return True

    if check('i need help'):
        return SimpleAnswer("Don't ask to ask, just ask your question")
    elif check('will someone help'):
        return SimpleAnswer("Don't ask to ask, just ask your question")
    elif check('i have a question'):
        return SimpleAnswer("Don't ask to ask, just ask your question")
    else:
        return None


def get_answer(query):
    sa = get_simple_anser(query)
    if sa is not None:
        return sa
    p = cdqa_pipeline.predict(query=query)
    r = Response(
        query, p[0], p[1], p[2], p[3]
    )
    return r


if __name__ == "__main__":
    while True:
        query = input('> ')
        prediction = get_answer(query)

        if prediction[3] < -2 and False:
            print("cdQA: Sorry, I don't know.")
        else:
            print('query: {}\n'.format(query))
            print('cdQA: {}'.format(prediction[0]))
            print('title: {}\n'.format(prediction[1]))
            print('paragraph: {}\n'.format(prediction[2]))
            print('confidence: {}'.format(prediction[3]))