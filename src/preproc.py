#%%
from datasets import load_dataset
import pandas as pd
import json
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string 
stop = stopwords.words('english')


def get_dataset():
    data = load_dataset("../../NLP_yahoo_questions/tools/yahoo_answers_topics/yahoo_answers_topics.py")
    df=pd.DataFrame(data['train'])
    df['text']=df['question_title']+df['question_content']+df['best_answer']
    return df

'''
Dataset({
    features: ['id', 'topic', 'question_title', 'question_content', 'best_answer'],
    num_rows: 1400000
})
'''

def get_topics():
    f = open('../../NLP_yahoo_questions/tools/yahoo_answers_topics/dataset_infos.json')
    f1    = json.load(f)
    f.close()
    topics=f1['yahoo_answers_topics']['features']['topic']['names']
    topics={k+1:v for k,v in enumerate(topics)}
    return topics


####preprocessing



def process_series(series, stop):
    series = series.str.lower().str.replace('[^\w\s]', ' ').apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return series


# df['cleanedtext']=process_series(df['text'],stop)
# df.to_csv('../../NLP_yahoo_questions/data/cleanedtrain.csv')
