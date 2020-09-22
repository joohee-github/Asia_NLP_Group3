import pandas as pd
import numpy as np
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from konlpy.tag import Okt


# 가상 sentence 

sentence = '신나는 아이돌 노래 추천해줘'


#-------------sentence 전처리--------------#

def music_preprocessing_sentence(sentence):
    
    okt= Okt()
    okt_tokens = okt.pos(sentence)
    stopPos = ['Josa','Verb']
    stopWord = '듣기 좋은 추천해줘 추천 같은 처럼 같이 들으면 듣고 싶어 풍 추천 적 법 때 노래 닐 나'
    stopWord = stopWord.split()

    word = []
    
    for tag in okt_tokens:
        if tag[1] not in stopPos:
            if tag[0] not in stopWord:
                word.append(tag[0])
    
    #print(word)
    
    return word


# ---------전체 데이터리스트에서 unique한 tag 집합만 가져옴---------- # 

def music_get_unique_x(file):

    data = pd.read_csv(file)
    features = data['tag'].tolist()
    
    # 중복값 제거
    unique_x = set(features)
    unique_x = list(unique_x)
    
    # 리스트값 셔플
    random.shuffle(unique_x)
    
    return unique_x


# ------------가상 선호 모델 설정-------------- #

def music_make_user_model(sentence):
    
    unique_x = music_get_unique_x('chatapp/ChatFramework/data/music/music_tag.csv')
    favor_label = np.zeros((len(unique_x)))
    sentence = music_preprocessing_sentence(sentence)

    for word in sentence:
        for i in range(len(unique_x)):
            if word in unique_x[i]:
                favor_label[i] += 1 

    max_class = int(np.max(favor_label))
    
    return unique_x, favor_label, max_class


#--------train data와 test data 나누기----------#

def music_split_data():
    
    total_data, total_label, max_class = music_make_user_model(sentence)
    
    train_data = total_data[:7450]
    test_data = total_data[7450:]

    train_label = total_label[:7450]
    test_label = total_label[7450:]

    return train_data, test_data, train_label, test_label



#-------------BoW형태 데이터 준비-----------------#

def music_prepare_data():
    
    train_data, test_data, train_label, test_label = music_split_data()
    max_words = 500
    
    t = Tokenizer(num_words = max_words) 
    t.fit_on_texts(train_data)
    X_train = t.texts_to_matrix(train_data, mode = 'count') 
    X_test = t.texts_to_matrix(test_data, mode = 'count') 
    
    return X_train, X_test, train_label, test_label, max_words

#------------BoW 형태의 train, test 데이터 준비------------------#

def music_set_data():
    
    max_class = music_make_user_model(sentence)[2]
    X_train, X_test, train_label, test_label, max_words = music_prepare_data()
    
    y_train = to_categorical(train_label, max_class+1) 
    y_test = to_categorical(test_label, max_class+1) 

    return X_train, X_test, y_train, y_test, max_class

#----------------모델 학습-----------------#

def music_fit_and_evaluate():
    
    X_train, X_test, y_train, y_test, max_class = music_set_data()
    max_words = music_prepare_data()[4]
    
    model = Sequential() 
    
    model.add(Dense(256, input_shape = (max_words,), activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(max_class+1, activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer= 'rmsprop',
                 metrics =['accuracy'])
    
    model.fit(X_train,y_train, batch_size = 128, epochs = 6, verbose = 1, validation_split = 0.1)
    results = model.evaluate(X_test, y_test, batch_size = 128, verbose = 0)
    
    # 정확도 확인
    print(results[1])
    
    return model 


#--------------예측 모델 BoW--------------#

def music_prediction():
    
    return_data = pd.read_csv('chatapp/ChatFramework/data/music/music_title.csv')
    return_feature = return_data['특징']
    t = Tokenizer(num_words = 500) 
    t.fit_on_texts(return_feature)
    return_feature = t.texts_to_matrix(return_feature, mode = 'count')
    
    
    return return_data, return_feature

#-------------모델 예측----------------#

def music_apply_predict():
    
    model = music_fit_and_evaluate()
    return_data, return_feature = music_prediction()    
    predict_value = model.predict(return_feature)
    
    predict_label = []
    
    for i in range(len(predict_value)):
        predict_label.append(np.argmax(predict_value[i]))
        
    return_data['label'] = predict_label
    
    return return_data
    
#------------대답하기-------------------#  

def music_return_to_page():
    
    return_data = music_apply_predict()
    new_df = return_data[return_data['label'] == return_data['label'].max()].reset_index(drop=True)
    df_elements = new_df.sample(n=1)
    df_elements = df_elements.to_dict(orient="record")[0]
    answer = '멋진 취향이네요! 오늘은 {}의 {}를 들어보세요.'.format(df_elements['artist'], df_elements['title'])
    
    return answer

#---------------총 실행--------------------#

def get_answer(_sentence):
       
    global sentence
    sentence = _sentence
    unique_x, favor_label, max_class = music_make_user_model(_sentence)
    answer = music_return_to_page() 

    return answer


# 예측
# get_answer(sentence)