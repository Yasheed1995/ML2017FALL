import numpy as np
import pandas as pd
import sys
import csv
import jieba
from opencc import OpenCC
from gensim.models import Word2Vec
from gensim.utils import tokenize, to_utf8
openCC = OpenCC('t2s')
EMBEDDING_DIM = int(sys.argv[1])
window = int(sys.argv[2])

workers = 8
itertation = 50
min_c = 1
model_save_path = 'w2v_model_simplify_weighted/dim'+str(EMBEDDING_DIM)+'_win'+str(window)+'_min' + str(min_c)+'_iter'+str(itertation)+'.bin'
outfile_path = 'result/dim'+str(EMBEDDING_DIM)+'_win'+str(window)+'_min' + str(min_c)+'_iter'+str(itertation)+ '_simplify.csv'
train_w2v = 1
a = int(sys.argv[3])

def readfile_train():
    sentences,data=[],[]
    for i in range(1,6):
        file_path='data/training_data/'+str(i)+'_train.txt'
        print ('Reading file from '+file_path+'...')
        f = open(file_path,'r',encoding='utf8')
        for row in f:
            row = row[:-1]
            row = openCC.convert(row)
            data.append(row)
            sentences.append(jieba.lcut(row, cut_all=False))
        f.close()
    return sentences,data

def readfile_test():
    f = open('data/testing_data.csv','r',encoding='utf8')
    next(f)
    question,option=[],[]
    for row in f:
        temp, q, o=row.strip().split(',')
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

sen,data=readfile_train()
print (sen[0])
sentences = []
for i in range(len(sen)-2):
    tmp = []
    tmp = sen[i]+sen[i+1]+sen[i+2]
    sentences.append(tmp)



# do weighted sum
        


print(len(sentences))
print('train word2vec ...')
if(train_w2v):
    w2v_model = Word2Vec(sentences,size = EMBEDDING_DIM,min_count=min_c,workers=8,sg=1,iter=itertation,window=window)
    w2v_model.save(model_save_path)
else:
    w2v_model = Word2Vec.load('w2v_model/model.bin')

print('word2vec save done ...')
print(w2v_model)
question,options = readfile_test()

print( len(question))
words = list(w2v_model.wv.vocab)

questions_vec = np.zeros((0,EMBEDDING_DIM))
ans = []
for i in range(len(question)):
	cnt = 0
	avg_emb = np.zeros((EMBEDDING_DIM,))
	for word in jieba.cut(question[i]):
		if word in w2v_model:
		    avg_emb += w2v_model[word]
                    vocab_obj = w2v.vocab[word]
                    avg_emb += w2v_model[word] * (a / (a+vocab_obj.count))
		    cnt += 1
	avg_emb /= cnt
	questions_vec = np.vstack((questions_vec,avg_emb))

print(questions_vec)
print(questions_vec.shape)

# predict similarity
for i in range(len(options)):
	max_idx = -1
	max_sim = -10
	for j in range(6):
		cnt = 0
		avg_emb = np.zeros((EMBEDDING_DIM,))
		for word in jieba.cut(options[i][j]):
			if word in w2v_model:
				avg_emb += w2v_model[word]
                                vocab_obj = w2v.vocab[word]
                                avg_emb += w2v_model[word] * (a / (a+vocab_obj.count))
				cnt += 1
		avg_emb /= cnt
		sim = np.dot(questions_vec[i], avg_emb) / np.linalg.norm(questions_vec[i]) / np.linalg.norm(avg_emb)
		if(sim > max_sim):
			max_idx = j
			max_sim = sim
	ans.append(max_idx)
print('predict done ...')

f = open(outfile_path,'w')
writer = csv.writer(f,delimiter=',',lineterminator='\n')
writer.writerow(["id","ans"])
for i in range(len(ans)):
    writer.writerow([i+1,str(ans[i])])
