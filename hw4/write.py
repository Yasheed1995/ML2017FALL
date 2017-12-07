
import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='Handle output.')
parser.add_argument('--test_y', dest='test_y',type=str,required=True)
args = parser.parse_args()
output_file="output.csv"
test_y=np.load(args.test_y)
#test_y=np.argmax(test_y,axis=1)
test_y = np.array([[1] if i[0] > 0.5 else [0] for i in test_y])
idx=np.array([[j for j in range(len(test_y))]]).T
print(test_y.shape)
print(idx.shape)
print (test_y)
test_y=np.hstack((idx,test_y)).astype(int)
#print(output.shape)
myoutput=pd.DataFrame(test_y,columns=["id","label"])
myoutput.to_csv(output_file,index=False)
'''
'''
