import sentencepiece as spm
import os

def saveSentence():
    # subword 학습을 위해 문장만 따로 저장
    with open('../data/train_tokenizer.txt', 'r', encoding='utf-8') as f:
        test_tokenizer = f.read().split('\n')
    print(test_tokenizer[:3])

    num_word_list = [len(sentence.split()) for sentence in test_tokenizer]
    print('\n코퍼스 평균/총 단어 갯수 : %.1f / %d' % (sum(num_word_list)/len(num_word_list), sum(num_word_list)))

def train():
    # spm_train --input=data/train_tokenizer.txt  --model_prefix=sentencepiece/sp --vocab_size=32000 character_coverage=1.0 --model_type="unigram"

    input_file = '../data/train_tokenizer.txt'
    vocab_size = 32000

    sp_model_root='sentencepiece'
    if not os.path.isdir(sp_model_root):
        os.mkdir(sp_model_root)

    sp_model_name = 'tokenizer_%d' % (vocab_size)
    sp_model_path = os.path.join(sp_model_root, sp_model_name)

    model_type = 'unigram'  # 학습할 모델 선택, unigram이 더 성능이 좋음'bpe'
    character_coverage  = 1.0  # 전체를 cover 하기 위해, default=0.9995
    user_defined_symbols = '[PAD],[UNK],[CLS],[SEP],[MASK],[BOS],[EOS],[UNK0],[UNK1],[UNK2],[UNK3],[UNK4],[UNK5],[UNK6],[UNK7],[UNK8],[UNK9],[unused0],[unused1],[unused2],[unused3],[unused4],[unused5],[unused6],[unused7],[unused8],[unused9],[unused10],[unused11],[unused12],[unused13],[unused14],[unused15],[unused16],[unused17],[unused18],[unused19],[unused20],[unused21],[unused22],[unused23],[unused24],[unused25],[unused26],[unused27],[unused28],[unused29],[unused30],[unused31],[unused32],[unused33],[unused34],[unused35],[unused36],[unused37],[unused38],[unused39],[unused40],[unused41],[unused42],[unused43],[unused44],[unused45],[unused46],[unused47],[unused48],[unused49],[unused50],[unused51],[unused52],[unused53],[unused54],[unused55],[unused56],[unused57],[unused58],[unused59],[unused60],[unused61],[unused62],[unused63],[unused64],[unused65],[unused66],[unused67],[unused68],[unused69],[unused70],[unused71],[unused72],[unused73],[unused74],[unused75],[unused76],[unused77],[unused78],[unused79],[unused80],[unused81],[unused82],[unused83],[unused84],[unused85],[unused86],[unused87],[unused88],[unused89],[unused90],[unused91],[unused92],[unused93],[unused94],[unused95],[unused96],[unused97],[unused98],[unused99]'

    input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --user_defined_symbols=%s --model_type=%s --character_coverage=%s'
    cmd = input_argument%(input_file, sp_model_path, vocab_size,user_defined_symbols, model_type, character_coverage)

    spm.SentencePieceTrainer.Train(cmd)
    print('train done')

    return sp_model_path

def split(sp_model_path):
    sp = spm.SentencePieceProcessor()
    sp.Load('{}.model'.format(sp_model_path))

    tokens = sp.encode_as_pieces('나는 오늘 아침밥을 먹었다.')
    ids = sp.encode_as_ids('나는 오늘 아침밥을 먹었다.')

    print(ids)
    print(tokens)

    tokens = sp.decode_pieces(tokens)
    ids = sp.decode_ids(ids)

    print(ids)
    print(tokens)