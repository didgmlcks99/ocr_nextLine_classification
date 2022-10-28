def recordInfo(fn, data):
    with open('../data/processed'+fn, 'w') as f:
        ls = []

        for key in data:
            target = key + ': ' + str(data[key])
            ls.append(target)
        
        f.write('\n'.join(ls))
    
    print('saved ch2idx to file!')