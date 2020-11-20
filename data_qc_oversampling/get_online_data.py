id2text = {}
with open('final_train.txt') as f:
    for line in f:
        line = line.strip().split('|,|')
        if line[0] not in id2text:
            id2text[line[0]] = [line[2]]
        else:
            if line[2] not in id2text[line[0]]:
                id2text[line[0]].append(line[2])

d_sorted = sorted(id2text.items(), key=lambda x:x[0])
with open('get_online_data.txt', 'w') as fw:
    for key, value in d_sorted:
        fw.write(key+'\n')
        for line in value:
            fw.write(line+'\n')