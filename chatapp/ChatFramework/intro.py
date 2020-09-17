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

#[웹툰,음악,영화]데이터 분류모델 로드, 해당 기능 단어사전데이터 로드
opening_model = keras.models.load_model('data/webtoon/opening_conv.h5')
opening_model_data = np.load('data/webtoon/opening_model_data.npy',allow_pickle='TRUE').item()
open_classes = np.load('data/webtoon/open_classes.npy',allow_pickle='TRUE').item()['file']
open_words = np.load('data/webtoon/open_words.npy',allow_pickle='TRUE').item()['file']

def clean_up_sentence(sentence,stop_word):
    """들어온 문장을 토큰화해서 리턴해주는 함수"""

    pos_result = twitter.pos(sentence, norm=True, stem=True)
    sentence_words = [lex for lex, pos in pos_result if lex not in stop_word]

    return sentence_words

def bow(sentence, words, show_details=False, stop_word = []):
    """문장과 단어책을 받아서 clean_up_sentence함수로 토큰화된 단어들을 받고 Bag of word생성"""

    sentence_words = clean_up_sentence(sentence,stop_word)
    print(sentence_words)

    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def predict_class(sentense,model,data_len,classes,words,stop_word=[]):
    """사용자에 입력에 따라 값(웹툰,음악,영화)을 예측하는 함수 리턴값은 예측 라벨과 각 라벨에 대한 확률값"""
    p = bow(sentense, words, stop_word=stop_word)
    p = np.array(p)
    p = p.reshape(1,data_len)
    predict_value = model.predict(p)
    class_value = classes[np.argmax(predict_value[0])]
    return class_value, predict_value

def _predict_class(sentence) :
     return predict_class(sentence,opening_model,opening_model_data['words'],open_classes,open_words)

def main_scenario():

    user_answer = input('어떤 콘텐츠를 추천해 드릴까요?')
    answer, _ = predict_class(user_answer,opening_model, opening_model_data['words'], open_classes, open_words)
    # predict_class함수 리턴 유형
    ## 1. 라벨 유형 : ['goodbye', 'hello', 'movie', 'music', 'service', 'thanks', 'webtoon']
    ## 2. 각 라벨에 대한 확률 값 : [[0.00151726 0.01952674 0.00542705 0.09988319 0.01905727 0.00411863 0.8504699 ]]
    # open_classes안에 클래스 목록들 저장되어 있음
    print(answer, '를 추천해드릴게요')