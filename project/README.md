# Machine Learning Final Project

Listen & Translate

## Getting Started

Given a Taiwanese audio signal, select the most possible Chinese translations from the given options. For more details, please refer to the lecture slides.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Evaluation Metric

The evaluation metric for this competition is accuracy. If there are m correct answers out of N questions, then the accuracy would simply be m / N.

## Running the tests

Explain how to run the automated tests for this system

## Data Files

Given Files:
  data.zip [Data]
  |＿ train.data
  |＿ train.caption
  |＿ test.data
  |＿ test.csv
  |＿ example_wav.zip


### Data Preprocessing

Use Kaldi to extract a series of MFCC feature vectors(39 dim) from audio signal.

### references

seq2seq

http://blog.csdn.net/jerr__y/article/details/53749693


(ML 2017) 

http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/RNN%20(v2).pdf
http://cyruschiu.github.io/2017/02/24/learning-Tensoflow-Seq2Seq-for-translate/


Hinge loss

http://blog.csdn.net/luo123n/article/details/48878759


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc

