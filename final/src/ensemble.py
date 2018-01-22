import numpy as np
import pandas as pd
import sys
import csv
import os
import jieba
from opencc import OpenCC
from gensim.models import Word2Vec
from gensim.utils import tokenize, to_utf8
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#jieba.set_dictionary('data/dict.txt')
#EMBEDDING_DIM = int(sys.argv[1])
#window = int(sys.argv[2])
workers = 8
itertation =50
min_c = 1
#model_save_path = 'w2v_model/dim'+str(EMBEDDING_DIM)+'_win'+str(window)+'_min' + str(min_c)+'_iter'+str(itertation)+'.bin'
#outfile_path = 'result/dim'+str(EMBEDDING_DIM)+'_win'+str(window)+'_min' + str(min_c)+'_iter'+str(itertation)+ '.csv'
#seq_len=25
#train_token=1
#train_w2v=0

a = int(sys.argv[1])

def readfile_test(testing_file_path):
    f = open(testing_file_path,'r',encoding='utf8')
    next(f)
    question,option=[],[]
    for row in f:
        temp, q, o=row.strip().split(',')
        q = openCC.convert(q)
        o = openCC.convert(o)
        q = q[2:]
        q = q.replace('A:', '')
        q = q.replace('B:', '')
        o1,o2,o3,o4,o5,o6 = o.split('\t')
        o1 = o1[2:]
        o2 = o2[2:]
        o3 = o3[2:]
        o4 = o4[2:]
        o5 = o5[2:]
        o6 = o6[2:]
        option.append([o1,o2,o3,o4,o5,o6])
        question.append(q)
    f.close()
    return question,option
openCC = OpenCC('t2s')
dirs = os.listdir('good_model')
print(dirs)

question,options = readfile_test(sys.argv[2])
print (question[0])

#words = list(w2v_model.wv.vocab)
#print(w2v_model.similarity('女','男'))
"""
questions_vec = np.zeros((0,EMBEDDING_DIM))

for i in range(len(question)):
	cnt = 0
	avg_emb = np.zeros((EMBEDDING_DIM,))
	for word in jieba.cut(question[i]):
		if word in w2v_model:
		    avg_emb += w2v_model[word]
		    cnt += 1
	avg_emb /= cnt
	questions_vec = np.vstack((questions_vec,avg_emb))

print(questions_vec)
print(questions_vec.shape)
"""
ans_list = np.zeros((5060,6))
for file in dirs:
    if (file[-1]!='n'):
        print (file)
        continue
    w2v_model=Word2Vec.load('good_model/'+str(file))
    EMBEDDING_DIM = w2v_model.vector_size
    print(EMBEDDING_DIM)
    questions_vec = np.zeros((0,EMBEDDING_DIM))

    for i in range(len(question)):
        cnt = 0
        avg_emb = np.zeros((EMBEDDING_DIM,))
        for word in jieba.cut(question[i]):
            if word in w2v_model:
                avg_emb += w2v_model[word]
                vocab_obj = w2v_model.wv.vocab[word]
                avg_emb += w2v_model[word] * (a / (a+vocab_obj.count))

                cnt += 1
        if cnt == 0:
            ans_list[i][0] += 5
        if cnt == 1:
            ans_list[i][0] += 1
        if cnt != 0:
            avg_emb /= cnt
        questions_vec = np.vstack((questions_vec,avg_emb))

    for i in range(len(options)):
        max_idx = -1
        max_sim = -10
        for j in range(6):
            cnt = 0
            avg_emb = np.zeros((EMBEDDING_DIM,))
            for word in jieba.cut(options[i][j]):
                if word in w2v_model:
                    avg_emb += w2v_model[word]
                    
                    vocab_obj = w2v_model.wv.vocab[word]
                    avg_emb += w2v_model[word] * (a / (a+vocab_obj.count))

                    cnt += 1

            if cnt == 0:
                ans_list[i][j] -= 1
            if cnt != 0:
                avg_emb /= cnt
            sim = np.dot(questions_vec[i], avg_emb) / np.linalg.norm(questions_vec[i]) / np.linalg.norm(avg_emb)
            if(sim > max_sim):
                max_idx = j
                max_sim = sim
        ans_list[i][max_idx] += 1

print(ans_list)
ans = []
for i in range(5060):
    ans.append(np.argmax(ans_list[i]))
print('predict done ...')
#csv_name = 'ensemble_result/ensemble_simplify_64_128_256_weighted_%s.csv'%str(a)
csv_name = sys.argv[3]
f = open(csv_name,'w')
writer = csv.writer(f,delimiter=',',lineterminator='\n')
writer.writerow(["id","ans"])
for i in range(len(ans)):
    writer.writerow([i+1,str(ans[i])])
