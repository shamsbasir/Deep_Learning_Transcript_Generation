# Deep Learning Transcript Generation

This work is a part of the third homework assignment for introduction to deep learning(CMU-11785) at [Class Link] (https://www.kaggle.com/c/11-785-fall-20-homework-3-slack). 

In this challenge, I will use a ResNET block to extract features and a bidirectional LSTM to combine the temporal features to make order-synchronoous phoneme predictions


## DATA 
Data can be download at [data link] (https://www.kaggle.com/c/11-785-fall-20-homework-3-slack/data)


## DEPENDENCIES

* python 3.6[python package](https://www.python.org/downloads/)
* torch [pytorch package] (https://github.com/pytorch/pytorch)
* numpy [numpy package] (https://numpy.org/install/)
* levenshtein [package] (https://pypi.org/project/python-Levenshtein/) 
* CTCBeamDecoder [package] (https://github.com/parlance/ctcdecode.git)
* pandas    [package link] (https://pandas.pydata.org/docs/getting_started/index.html)


## MODEL ARCHITECTURE
```
------------------------------------------------------------------------
data---> ResNET block --> Bidirectional LSTM --> Linear --> output	  |
------------------------------------------------------------------------

SpeechToPhoneme(
  (conv1): Conv1d(13, 128, kernel_size=(1,), stride=(1,), bias=False)
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv1d(128, 256, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
  (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
  (bn3): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (shortcut): Sequential(
    (0): Conv1d(13, 256, kernel_size=(3,), stride=(1,), padding=(1,))
    (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (rnn): LSTM(256, 512, num_layers=4, batch_first=True, dropout=0.2, bidirectional=True)
  (linear): Linear(in_features=1024, out_features=512, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
  (output): Linear(in_features=512, out_features=42, bias=True)
)
```

## DIRECTORY STRUCTURE
``
|   README.txt
|   submission.csv
|   hw3_p2.ipynb 
``

## HYPER-PARAMATERS 

* batch_size : 64 for training
* batch_size : 32 for testing and validation 
* num_workers: 4


* Beam Width : 10 for validation
* Beam Width :100 for testing

 
## Optimzers

* Adam 
* lr 	     : 5e-4
* wd 	     : 5e-5


## Learning Rate Scheduler 

* ReduceLROnPlateau   # parameters can be found on main.ipyn or Classification.ipyn
* lr decay   : 0.85
* patience   : 0
* threshold  : 0.5


### TRAINING
* just run all the cells 
* in case Cuda memory error, Restart it from where u finished training 

### TESTING
* pick the best model and test
* run the cells for the rest of it 

## Questions?
shamsbasir@gmail.com

