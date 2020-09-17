############################################ 라이브러리 로드 #########################################

#Json파일 로드와 저장을 위한 라이브러리
import json
import pickle
import random

#자연어 토큰화를 위한 라이브러리
from konlpy.tag import Okt
twitter = Okt()

#데이터 로드와 모델 사용을 위한 라이브러리
from keras.models import Sequential
import keras
import tensorflow as tf
import numpy as np
import pandas as pd

#코사인 유사도 계산을 위한 통계 라이브러리
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

######################################################################################################


############################################ 데이터 로드 ##############################################

#웹툰 데이터 로드
webtoon_data = np.load('data/webtoon/web_toon_data.npy',allow_pickle='TRUE').item()

#웹툰 장르 분류모델 로드, 해당 기능 단어사전데이터 로드
gerne_model = keras.models.load_model('data/webtoon/gerne_conv.h5')
gerne_model_data = np.load('data/webtoon/gerne_model_data.npy',allow_pickle='TRUE').item()
gerne_classes = np.load('data/webtoon/gerne_classes.npy',allow_pickle='TRUE').item()['file']
gerne_words = np.load('data/webtoon/gerne_words.npy',allow_pickle='TRUE').item()['file']

#######################################################################################################


############################################ 필요한 변수 및 데이터 정리 #################################

#웹툰 장르 구별에서 사용자 입력 단어 중 불필요한 단어 제거
stop_word = ['웹툰','추천','내용','이야기','전개']

#웹툰 데이터 정제(원본데이터 -> data변수로 필요한 데이터만 추가)
data = []
for work in webtoon_data:
    title = webtoon_data[work]['title']
    category = webtoon_data[work]['category']
    summary = webtoon_data[work]['summary']
    gerne = webtoon_data[work]['gerne']
    data.append({
        'title':title,
        'category':category,
        'gerne':gerne,
        'summary':category[0] +' '+ category[1] +' '+ summary
    })
data = pd.DataFrame(data)

#######################################################################################################


############################################ 함수 정의 #################################################

def clean_up_sentence(sentence,stop_word):
    """문장 토큰화"""

    pos_result = twitter.pos(sentence, norm=True, stem=True)
    sentence_words = [lex for lex, pos in pos_result if lex not in stop_word]

    return sentence_words

def bow(sentence, words, show_details=False, stop_word = []):
    """clean_up_sentence에서 토큰화된 단어들을 받아서 Bag of word생성"""

    sentence_words = clean_up_sentence(sentence,stop_word)
    print(sentence_words)

    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def add_user_sentence(user_text,answer):
    """사용자가 원하는 내용을 받아서 데이터셋에 추가한 후 추가한 데이터셋 반환"""
    if answer[0] == answer[1]:
        newtf = data['gerne'] == answer[0]
        user_data= data[newtf]
    else :
        newtf = data['gerne'] == answer[0]
        newtf2 = data['gerne'] == answer[1]
        user_data= data[newtf].append(data[newtf2])
        
    user_data = user_data.append({'title':'UserData','category':'','gerne':'','summary':user_text},ignore_index=True)
    return user_data

def cosine_similarity(data):
    """데이터셋을 받아서 코사인 유사도와 단어 인덱스 사전 반환"""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['summary'])
    print('{}개의 데이터셋과 {}개의 단어 구성'.format(tfidf_matrix.shape[0],tfidf_matrix.shape[1]))
    
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(data.index, index=data['title']).drop_duplicates()
    
    return cosine_sim, indices

def get_recommendations(data, indices, cosine_sim, title="UserData"):
    """사용자가 입력한 데이터를 받아서 계산한 코사인 유사도 표를 통해 유사한 콘텐츠 10개 리턴"""
    # 선택한 웹툰의 타이틀로부터 해당되는 인덱스를 받아옵니다. 이제 선택한 웹툰를 가지고 연산할 수 있습니다.
    idx = indices[title]

    # 모든 웹툰에 대해서 해당 웹툰과의 유사도를 구합니다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 웹툰들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 웹툰을 받아옵니다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 웹툰의 인덱스를 받아옵니다.
    movie_indices = [i[0] for i in sim_scores]

    # 가장 유사한 10개의 웹툰의 제목을 데이터 프레임으로 리턴합니다.
    return data['title'].iloc[movie_indices]

def predict_class(sentense,model,data_len,classes,words,stop_word=[]):
    """사용자에 입력에 따라 값을 예측하는 함수"""
    p = bow(sentense, words, stop_word=stop_word)
    p = np.array(p)
    p = p.reshape(1,data_len)
    predict_value = model.predict(p)
    class_value = classes[np.argmax(predict_value[0])]
    return class_value, predict_value

def _predict_class(sentence) :
     return predict_class(sentence,gerne_model,gerne_model_data['words'],gerne_classes,gerne_words,stop_word=stop_word)

#######################################################################################################


############################################ 시나리오 ##################################################

def main_scenario():
    context = ''
    gerne = []
 
    user_answer = input('어떤 장르의 웹툰을 추천해 드릴까요?')
    answer, _ = predict_class(user_answer,gerne_model,gerne_model_data['words'],gerne_classes,gerne_words,stop_word=stop_word)
    context += ' '+user_answer
    gerne.append(answer)
    gerne.append(answer)
    print(answer, '를 추천해드릴게요')
    
    user_answer = input('어떤 스토리의 웹툰을 추천해 드릴까요?')
    answer, _ = predict_class(user_answer,gerne_model,gerne_model_data['words'],gerne_classes,gerne_words,stop_word=stop_word)
    # gerne.append(answer)
    print(answer, '를 추천해드릴게요')
    
    context += ' '+user_answer
    data_set = add_user_sentence(context,gerne)
    cosine_sim, idx_dict = cosine_similarity(data_set)
    print(get_recommendations(data_set, idx_dict, cosine_sim))
#######################################################################################################