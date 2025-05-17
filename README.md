# DeepScalp

MOEX scalping trade bot based on deep learning model.

Uses adopted variant of [LSHASH](https://github.com/kayzhu/LSHash) to speed up similarity measurement of trading data samples.

# Prerequisites

* https://numpy.org/
* https://tinkoff.github.io/invest-python/
* https://pypi.org/project/dearpygui/
* https://pypi.org/project/joblib/
* https://pypi.org/project/win10toast/
* https://pytorch.org/get-started/locally/

# Installation

* pip install numpy
* pip install tinkoff-investments
* pip install dearpygui
* pip install joblib
* pip install win10toast
* pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Access token

* The scripts requires T-Bank access token in order to integrate with its API. 
* The token should be provided in environment variable TK_TOKEN on the local machine.

# Gathering trading data

* To train the models you will need to gather enough of the trading data using TkDataGatherLoop.bat
* The script feeding from the live stock market exchange (MOEX) and outputs a set of files representing streams of trading orders and operations relevant to certain shares.

An example of the gathered tading data:

![python_ogQfLvLvfv](https://github.com/user-attachments/assets/c08fddd4-59ab-43b7-86d5-baa12dc2ee63)

# Training autoencoders

* The forecasting model uses compressed representation of orderbooks and last trades distribution samples.
* The aforementioned compressed representation is a learnable models based on autoencoders.
* To train those autoencoders you need to preprocess the gathered trading data using TkPreprocessAutoencoderData.py
* Upon completion of preprocessing data, you can launch TkTrainAutoencoders. It will display feedback regarding the training process, so you can decide for yourself when it should be stopped.

  ![python_6Lbg4oOwMS](https://github.com/user-attachments/assets/3852f933-f45c-472f-8198-a7d58ba469ae)

# Training time series forecasting model

* With trained autoencoder models we can prepare training data for time series prediction model using TkPreprocessTimeSeriesData.py.
* Upon completion of preprocessing data, it is all ready for launching TkTrainTimeSeries.py and training of the forecasting model. The training script will display feedback regarding the training process as well, it is just take significantly more time than training of the autoencoders.

![The time series model trained and reinforced with novel data.](https://github.com/user-attachments/assets/a8341555-1969-49cb-8616-db911d7e23bf)
![The time series model trained and reinforced with novel data.](https://github.com/user-attachments/assets/b2d6351d-895e-400e-9182-e490e0b793a6)


# Using forecasting service

In trading mode it is required to run both:
* TkDataGatherLoop.bat - to gather realtime trading data
* and TkForecastingService.py - to predict the price movement based on the gathered realtime trading data

Depending on the model parameters TkForecastingService.py would need for the trading data to accumulate for some time before starting predicting price movement.

https://github.com/user-attachments/assets/a25c5d02-5546-4b14-b827-2fd714ebbd98

