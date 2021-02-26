import pandas as pd
import numpy as np
import os
from keras import backend as K
from datetime import datetime, timedelta


class BaseForecast:
    """
    Base Forecast class used to build forecasting models

    Attributes:
        load_df (DataFrame): Load Profile Data
        train_frequency (str) [Optional] : Frequency of data for model 
                                        training (ex:"15min" or "60min")
    """

    def __init__(self, df, train_frequency="60min"):
        # initialize blank variables to be set later
        self.model = None
        self.train = None
        self.test = None
        self.scaler = None
        self.train_frequency = train_frequency
        self.multiple = 1
        if self.train_frequency == "15min":
            self.multiple = 4
        self.df = df
        self.frequency = pd.infer_freq(self.df.index)
        if not self.frequency:
            self.frequency = "15min"  # most cases it's 15min
        if "T" in self.frequency:  # standardize freq representation
            self.frequency = self.frequency.replace("T", "min")


    def fit(self):
        pass

    def process(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass


    def split_dataset(self, data, days, train_test_split, number_of_days_to_test, number_of_days_to_train):
        days = len(data) / (24 * self.multiple)
        if number_of_days_to_test:
            train_test_split_days = days - number_of_days_to_test
        elif number_of_days_to_train:
            train_test_split_days = number_of_days_to_train
        else:       
            train_test_split_days = int(train_test_split * days)
        split_at = int(train_test_split_days * 24 * self.multiple)  # 24 hours in a day
        # print(split_at)
        train, test = data[:split_at], data[split_at:]
        train = np.array(np.split(train, len(train) / (24 * self.multiple)))
        test = np.array(np.split(test, len(test) / (24 * self.multiple)))
        # print('train shape is:', train.shape)
        # print('test shape is:', test.shape)

        return train, test, split_at

    # summarize scores (includes overall rmse/mbe/mae score and scores for every hour of next 24 hours)
    def summarize_scores(self, name, score, scores):
        s_scores = ", ".join(["%.1f" % s for s in scores])
        print("%s: [%.3f] %s" % (name, score, s_scores))

