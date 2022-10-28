import string
import pandas as pd
import random
import numpy as np

# cuts sentence at random index
# in the standard of words
def cut(sentence):
    words = sentence.split()
    lim = len(words)

    if lim == 1: cut = lim
    else: cut = random.randint(1, lim-1)

    cut = lim // 2

    cut_first = ' '.join(words[:cut])
    cut_second = ' '.join(words[cut:])

    if len(cut_first) != 1:
        # 단어에 마지막 한 글자만 second으로 넘겨줌
        cut_second = cut_first[len(cut_first)-1:] + ' ' + cut_second
        cut_first = cut_first[:len(cut_first)-1]

    return cut_first, cut_second

def getInitData():
    
    length_list = []
    tot_data = []

    # tot_data += getChatBot(length_list)
    # print('got ChatBot data!')

    tot_data += getKCC(length_list)
    print('got KCC data!')

    # tot_data += getKo(length_list)
    # print('got KO data!')

    random.shuffle(tot_data)
    print('shuffled data!')

    df = pd.DataFrame(tot_data, columns=['first', 'second', 'label'])
    print('saved to dataframe!')

    length_avg = sum(length_list) / len(length_list)

    with open('../data/processed/relation/avg_length', 'w') as f:
        f.write(str(int(length_avg)))
        print('wrote length avg: ' + str(int(length_avg)))

    return df

def getChatBot(length_list):
    chatBot_df = pd.read_csv('../data/raw/ChatbotData.csv').sample(frac=1)

    total_data = []

    mixed_data = []
    mixed_first = []
    mixed_second = []
    related_data = []

    mixed_num =  4000
    related_num =  4000

    data_cnt = 0
    # loop through data
    for index, row in chatBot_df.iterrows():

        # save question as first and answer as second
        first = row['Q']
        second = row['A']

        if len(first.split()) <= 2: continue

        # remove full stop points at the end of each sentence
        first = first.translate(str.maketrans('', '', string.punctuation))

        if data_cnt < mixed_num:

            # split sentence
            first, second = cut(first)
            # add to data that will be mixed later on as 0
            # mixed_data.append([first, second, 0])
            mixed_first.append(first)
            mixed_second.append(second) 
        else:
            # split sentence
            first, second = cut(first)
            # add to data that will be related as 1
            related_data.append([first, second, 1])

        # save the length for each parts of sentence to calculate average length
        # in order to reduce rnn encoder size 
        length_list.append(len(first))
        length_list.append(len(second))

        # keep track of data on each related and not related
        data_cnt += 1

        # end when all 6000 not and 3000 not related have been accumulated
        if data_cnt == (mixed_num+related_num): break
    
    # mix the second sentence data from 6000 mixed_data
    random.shuffle(mixed_second)
    mixed_label = [0] * mixed_num

    # set for mixed data
    for i in range(mixed_num):
        mixed_data.append([mixed_first[i], mixed_second[i], mixed_label[i]])

    # move 6000 not and 3000 related data to total_data
    total_data += mixed_data
    total_data += related_data

    print('data count on chatbot: '+str(data_cnt))

    return total_data

def getKCC(length_list):
    kccN_txt = '../data/raw/KCC150_Korean_sentences_UTF8.txt'
    kccQ_txt = '../data/raw/KCCq28_Korean_sentences_UTF8_v2.txt'

    kccN_df = pd.read_csv(kccN_txt, sep='\t', header=None)
    kccQ_df = pd.read_csv(kccQ_txt, sep='\t', header=None)
    
    total_data = []

    mixed_data = []
    mixed_first = []
    mixed_second = []
    related_data = []

    # total 22000
    al = 2000000 
    mixed_num = int(al/4)
    related_num = int(al/4)

    data_cnt = 0
    for index, row in kccN_df.iterrows():

        sentence = row[0]

        if len(sentence.split()) <= 9: continue

        # remove full stop points at the end of each sentence
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))

        if data_cnt < mixed_num:

            # split sentence
            first, second = cut(sentence)

            # add to data that will be mixed later on as 0
            # mixed_data.append([first, second, 0])
            mixed_first.append(first)
            mixed_second.append(second) 
        else:
            # split sentence
            first, second = cut(sentence)

            # add to data that will be related as 1
            related_data.append([first, second, 1])

        # save the length for each parts of sentence to calculate average length
        # in order to reduce rnn encoder size 
        length_list.append(len(first))
        length_list.append(len(second))

        # keep track of data on each related and not related
        data_cnt += 1

        # end when all 3000 not and 1500 not related have been accumulated
        if data_cnt == (mixed_num+related_num): break
    
    # mix the second sentence data from 6000 mixed_data
    random.shuffle(mixed_second)
    mixed_label = [0] * mixed_num

    # set for mixed data
    for i in range(mixed_num):
        mixed_data.append([mixed_first[i], mixed_second[i], mixed_label[i]])

    # move 6000 not and 3000 related data to total_data
    total_data += mixed_data
    total_data += related_data

    print('data count on kccN: '+str(data_cnt))

    
    
    mixed_data = []
    mixed_first = []
    mixed_second = []
    related_data = []

    data_cnt = 0
    for index, row in kccQ_df.iterrows():

        sentence = row[0]

        if len(sentence.split()) <= 2: continue

        # remove full stop points at the end of each sentence
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))

        if data_cnt < mixed_num:

            # split sentence
            first, second = cut(sentence)
            # add to data that will be mixed later on as 0
            # mixed_data.append([first, second, 0])
            mixed_first.append(first)
            mixed_second.append(second) 
        else:
            # split sentence
            first, second = cut(sentence)
            # add to data that will be related as 1
            related_data.append([first, second, 1])


        # save the length for each parts of sentence to calculate average length
        # in order to reduce rnn encoder size 
        length_list.append(len(first))
        length_list.append(len(second))

        # keep track of data on each related and not related
        data_cnt += 1

        # end when all 3000 not and 1500 not related have been accumulated
        if data_cnt == (mixed_num+related_num): break
    
    # mix the second sentence data from 6000 mixed_data
    random.shuffle(mixed_second)
    mixed_label = [0] * mixed_num

    # set for mixed data
    for i in range(mixed_num):
        mixed_data.append([mixed_first[i], mixed_second[i], mixed_label[i]])

    # move 6000 not and 3000 related data to total_data
    total_data += mixed_data
    total_data += related_data
    
    print('data count on kccQ: '+str(data_cnt))

    return total_data

def getKo(length_list):
    ko_df = pd.read_csv('../data/raw/Ko_persona_train_corrected.csv', low_memory=False).sample(frac=1)

    total_data = []

    mixed_data = []
    mixed_first = []
    mixed_second = []
    related_data = []

    mixed_num = 5000
    related_num = 5000

    data_cnt = 0
    for index, row in ko_df.iterrows():
        
        first = row['dialogue/0/1']
        second = row['dialogue/1/0']

        if len(first.split()) <= 2: continue

        # remove full stop points at the end of each sentence
        first = first.translate(str.maketrans('', '', string.punctuation))

        if data_cnt >= related_num:
            first, second = cut(first)

            # add to data that will be mixed later on as 0
            # mixed_data.append([first, second, 0])
            mixed_first.append(first)
            mixed_second.append(second) 
        else:
            # split sentence
            first, second = cut(first)
            # add to data that will be related as 1
            related_data.append([first, second, 1])

        # save the length for each parts of sentence to calculate average length
        # in order to reduce rnn encoder size 
        length_list.append(len(first))
        length_list.append(len(second))

        # keep track of data on each related and not related
        data_cnt += 1

        # end when all 6000 not and 3000 not related have been accumulated
        if data_cnt == (mixed_num+related_num): break
    
    # mix the second sentence data from 6000 mixed_data
    random.shuffle(mixed_second)
    mixed_label = [0] * mixed_num


    # mixed_first_len = len(mixed_first)
    # if (mixed_first_len < mixed_num):
    #     for i in range(mixed_num - mixed_first_len):
    #         mixed_first.append(mixed_first[i])

    # print('added data on mixed_first Ko: ' + str(mixed_first_len))

    # mixed_second_len = len(mixed_second)
    # if (mixed_second_len < mixed_num):
    #     for i in range(mixed_num - mixed_second_len):
    #         mixed_second.append(mixed_second[i])

    # print('added data on mixed_second Ko: ' + str(mixed_second_len))

    # related_data_len = len(related_data)
    # if (related_data_len < related_num):
    #     for i in range(related_num - related_data_len):
    #         related_data.append(related_data[i])

    # print('added data on related Ko: ' + str(related_data_len))


    # set for mixed data
    for i in range(mixed_num):
        mixed_data.append([mixed_first[i], mixed_second[i], mixed_label[i]])

    # move 6000 not and 3000 related data to total_data
    total_data += mixed_data
    total_data += related_data

    print('data count on ko: '+str(data_cnt))

    return total_data

def mk_initData(df):
    
    # save dataframe created from 3 korean corpus processed to data file
    df.to_csv("../data/processed/relation/data", index=False, header=False)

    sentences_list = []
    label_list = []

    for index, row in df.iterrows():
        sentences_list.append(row['first'])
        sentences_list.append(row['second'])

        label_list.append(row['label'])
    
    label_df = pd.DataFrame(label_list, columns=['label'])

    # different way to save sentence list to differentiate between sentence and label
    to_file(sentences_list, "../data/processed/relation/sentence")

    label_df.to_csv("../data/processed/relation/label", index=False, header=False)

def to_file(ls, fn):
    with open(fn, 'w') as f:
        f.write('\n'.join(ls))

def getData():

    first_sentences = []
    second_sentences = []
    labels = []

    avg_length = 0

    with open('../data/processed/relation/avg_length', 'r') as f:
        avg_length = int(f.read()) // 5
        print('avg length / 5: ' + str(avg_length))
    
    with open('../data/processed/relation/sentence', 'r') as f:
        sentences = f.read().splitlines()

        for i in range(0, len(sentences), 2):

            first = sentences[i]
            second = sentences[i+1]
            
            if len(first) > avg_length:
                first_sentences.append(str(first[-avg_length:]))
            else:
                first_sentences.append(str(first))

            # first_sentences.append(str(first))

            if len(second) > avg_length:
                second_sentences.append(str(second[:avg_length]))
            else:
                second_sentences.append(str(second))

            # second_sentences.append(str(second))



    with open('../data/processed/relation/label', 'r') as f:
        ls = f.read().splitlines()

        for d in ls:
            labels.append(int(d))
    
    return np.array(first_sentences), np.array(second_sentences), np.array(labels)