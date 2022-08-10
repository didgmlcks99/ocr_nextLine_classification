import pandas as pd
import random

def cut(sentence):
    words = sentence.split()
    lim = len(words)

    if lim == 1: cut = lim
    else: cut = random.randint(1, lim-1)

    cut_first = ' '.join(words[:cut])
    cut_second = ' '.join(words[cut:])

    return cut_first, cut_second

def getData():
    tot_data = []
    
    tot_data += getChatBot()

    return tot_data


def getChatBot():
    chatBot_df = pd.read_csv('../data/raw/ChatbotData.csv')
    
    data_cnt = 0
    class_cnt = 0
    cut_cnt = 0
    label = 0

    total_data = []

    for index, row in chatBot_df.iterrows():
        
        first = row['Q']
        second = row['A']

        if class_cnt < 5000:
            label = 0

            if cut_cnt < 2500:
                c_first, c_second = cut(first)
                
                first = c_first
                second = ' '.join([c_second, second])
            else:
                c_first, c_second = cut(second)

                first = ' '.join([first, c_first])
                second = c_second
        else: label = 1

        total_data.append([first, second, label])

        data_cnt += 1
        class_cnt += 1
        cut_cnt += 1

        if data_cnt == 10000: break

    return total_data