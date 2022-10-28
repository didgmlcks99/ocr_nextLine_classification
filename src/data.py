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

    return cut_first, cut_second

def makeData(class_num, class_cnt, cut_cnt, first, second):

    # first 5000 is for cut sentences
    if class_cnt < class_num:
        label = 0

        # first 2500 is for cut sentence on first
        if cut_cnt < (class_num//2):
            c_first, c_second = cut(first)
            
            first = c_first
            second = ' '.join([c_second, second])

            cut_cnt += 1
        else:
            c_first, c_second = cut(second)

            first = ' '.join([first, c_first])
            second = c_second
        
        class_cnt += 1
    # next 5000 has no cuts on sentence
    else: label = 1

    return [first, second, label], class_cnt, cut_cnt

def getInitData():

    length_list = []
    tot_data = []
    
    # tot_data += getChatBot()
    # print('got ChatBot data!')

    tot_data += getKCC(length_list)
    print('got KCC data!')

    # tot_data += getKo()
    # print('got KO data!')

    random.shuffle(tot_data)
    print('shuffled data!')

    df = pd.DataFrame(tot_data, columns=['first', 'second', 'label'])
    print('saved to dataframe!')

    length_avg = sum(length_list) / len(length_list)

    with open('../data/processed/split/avg_length', 'w') as f:
        f.write(str(int(length_avf)))
        print('wrote length avg: ' + str(int(length_avg)))

    return df


def getChatBot():
    chatBot_df = pd.read_csv('../data/raw/ChatbotData.csv').sample(frac=1)
    
    data_cnt = 0
    class_cnt = 0
    cut_cnt = 0

    total_data = []

    # loop through data
    for index, row in chatBot_df.iterrows():
        
        # save question as first and answer as second
        first = row['Q']
        second = row['A']

        if len(first.split()) == 1 or len(second.split()) == 1: continue

        # regarding the amount of data for each class,
        # make data: list of first, second, label 
        cuts = makeData(class_cnt, cut_cnt, first, second)

        total_data.append(cuts)

        # keep track of first 5000 as cut and next 5000 as non-cut
        class_cnt += 1
        cut_cnt += 1
        data_cnt += 1

        if data_cnt == 10000: break

    return total_data

def getKCC(length_list):
    kccN_txt = '../data/raw/KCC150_Korean_sentences_UTF8.txt'
    kccQ_txt = '../data/raw/KCCq28_Korean_sentences_UTF8_v2.txt'

    kccN_df = pd.read_csv(kccN_txt, sep='\t', header=None)
    kccQ_df = pd.read_csv(kccQ_txt, sep='\t', header=None)
    
    sentence_list = []

    data_cnt = 0

    al = 30000

    for index, row in kccN_df.iterrows():

        sentence = row[0]

        if len(sentence.split()) <= 9: continue

        sentence_list.append(sentence)

        data_cnt += 1

        if data_cnt == al: break
    
    print('data count on kccN: ' + str(data_cnt))
    

    data_cnt = 0
    point_cnt = 0

    for index, row in kccQ_df.iterrows():

        sentence = row[0]

        if len(sentence.split()) <= 9: continue

        sentence_list.append(sentence)

        data_cnt += 1

        if data_cnt == al: break

    print('data count on kccQ: ' + str(data_cnt))
    

    random.shuffle(sentence_list)


    class_cnt = 0
    cut_cnt = 0

    total_data = []

    for i in range(0, len(sentence_list), 2):
        
        first = sentence_list[i]
        second = sentence_list[i+1]

        length_list.append(len(first))
        length_list.append(len(second))

        cuts, class_cnt, cut_cnt = makeData((al//2), class_cnt, cut_cnt, first, second)

        total_data.append(cuts)
    
    return total_data

def getKo():
    chatBot_df = pd.read_csv('../data/raw/Ko_persona_train_corrected.csv', low_memory=False).sample(frac=1)
    
    data_cnt = 0
    class_cnt = 0
    cut_cnt = 0

    total_data = []

    for index, row in chatBot_df.iterrows():
        
        first = row['dialogue/0/1']
        second = row['dialogue/1/0']

        if len(first.split()) == 1 or len(second.split()) == 1: continue

        cuts = makeData(class_cnt, cut_cnt, first, second)

        total_data.append(cuts)

        class_cnt += 1
        cut_cnt += 1
        data_cnt += 1

        if data_cnt == 10000: break

    return total_data

def mk_initData(df, rm_gudu):
    # df.to_csv("../data/processed/data.csv", index=False, header=False)
    # df.to_excel('../data/processed/data.xlsx', index=False, header=False, sheet_name='sheet1')
    df.to_csv("../data/processed/split/data", index=False, header=False)

    # sentences_df = df[['first', 'second']]
    # label_df = df['label']

    sentences_list = []
    label_list = []

    # removing puctuation marks
    if rm_gudu == 1:
        for index, row in df.iterrows():

            first = row['first']
            second = row['second']

            first = first.translate(str.makeTrans('', '', string.punctuation))
            second = second.translate(str.makeTrans('', '', string.punctuation))

            # if row['first'][len(row['first'])-1] in ['.', '?', '!', ',', ';', ':']:
            #     first_sentence = row['first'][:len(row['first'])-1]
            # else:
            #     first_sentence = row['first']

            # if row['second'][len(row['second'])-1] in ['.', '?', '!', ',', ';', ':']:
            #     second_sentence = row['second'][:len(row['second'])-1]
            # else:
            #     second_sentence = row['second']

            sentences_list.append(first)
            sentences_list.append(second)
            
            label_list.append(row['label'])
    # no removing punctuation marks
    else:
        for index, row in df.iterrows():
            sentences_list.append(row['first'])
            sentences_list.append(row['second'])

            label_list.append(row['label'])
    
    label_df = pd.DataFrame(label_list, columns=['label'])

    if rm_gudu == 1:
        to_file(sentences_list, "../data/processed/split/sentence_nogudu")
    else:
        to_file(sentences_list, "../data/processed/split/sentence_yesgudu")

    label_df.to_csv("../data/processed/split/label", index=False, header=False)

def to_file(ls, fn):

    with open(fn, 'w') as f:
        f.write('\n'.join(ls))

def getData(rm_gudu):
    
    first = []
    second = []
    labels = []

    with open('../data/processed/split/avg_length', 'r') as f:
        avg_length = int(f.read()) // 4
        print('avg length / 4: ' + str(avg_length))
    
    # sentence without puctuation
    if rm_gudu == 1:
        with open('../data/processed/split/sentence_nogudu', 'r') as f:
            sentences = f.read().splitlines()

            for i in range(0, len(sentences), 2):
                
                first.append(str(sentences[i]))
                second.append(str(sentences[i+1]))
    else:
        with open('../data/processed/split/sentence_yesgudu', 'r') as f:
            sentences = f.read().splitlines()

            for i in range(0, len(sentences), 2):
                
                first.append(str(sentences[i]))
                second.append(str(sentences[i+1]))
    

    with open('../data/processed/split/label', 'r') as f:
        ls = f.read().splitlines()

        for d in ls:
            labels.append(int(d))
    
    return np.array(first), np.array(second), np.array(labels)

def getCh2idx():

    ch2idx = {}

    with open('../records/ch2idx', 'r') as f:
        data = f.read().splitlines()

        for d in data:
            r = d.split(': ')
            c = r[0]
            i = int(r[1])

            ch2idx[c] = i
    
    print('successfully got ch2idx!')

    return ch2idx