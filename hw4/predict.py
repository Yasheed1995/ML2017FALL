import sys
import numpy as np
import pandas as pd
test_y = np.load('save/t.npy')
print (test_y.shape)
idx = np.array([[j for j in range(len(test_y))]]).T
print (idx.shape)
test_y = [[1] if i[0] < i[1] else [0] for i in test_y]
test_y = np.hstack((idx, test_y)).astype(int)
output = pd.DataFrame(test_y, columns=['id', 'label'])
print (output)
output.to_csv(sys.argv[1], index=False)
