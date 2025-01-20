
# DeepScalp

MOEX scalping trade bot based on deep learning model.

Uses adopted variant of [LSHASH](https://github.com/kayzhu/LSHash) to speed up similarity measurement of trading data samples.

# Prerequisites

* https://numpy.org/
* https://tinkoff.github.io/invest-python/
* https://pypi.org/project/dearpygui/
* https://pypi.org/project/joblib/

# Installation

* pip install numpy
* pip install tinkoff-investments
* pip install dearpygui
* pip install joblib

# Access token

* The scripts requires T-Bank access token in order to integrate with API. 
* The token should be provided in environment variable TK_TOKEN on the local machine.

# Gathering trading data

* To train the models you will need to gather enough of the trading data using TkDataGatherLoop.bat
* The script feeding from the live stock market exchange (MOEX) and outputs a set of files representing streams of trading orders and operations relevant to certain shares.

An example of the gathered tading data:

![python_y3DZ4i6pbr](https://github.com/user-attachments/assets/509091af-7e51-4e83-8aaa-c364a2a04b98)
