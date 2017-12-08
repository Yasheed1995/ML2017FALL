import matplotlib.pyplot as plt
import pandas as pd
history=pd.read_csv('save/dict_history.csv', 'r')
print (history.keys())
#plt.plot([0.75,0.82,0.84,0.84,0.845,0.86])
#plt.plot([0.73,0.74,0.73,0.72,0.715,0.713])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_acc.png')
# summarize history for loss
plt.plot([3.01,0.43,0.40,0.38,0.35,0.33])
plt.plot([0.56,0.52,0.55,0.51,0.50,0.49])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_loss.png')
