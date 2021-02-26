from math import sqrt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dropout
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.metrics import mean_squared_error
import sklearn
import keras
from sklearn.preprocessing import MinMaxScaler
import os
import shutil
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.models import load_model as keras_load_model

from dateutil.relativedelta import relativedelta

from esap_forecast.models.forecast import BaseForecast


class RNN(BaseForecast):
    def __init__(self, *args, forecast_type=None, is_yearly_forecasts=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = "rnn_model"
        self.forecast_type = forecast_type
        self.is_yearly_forecasts = is_yearly_forecasts


    #Split the dataset, builds the model (tunes it), fits the training data to the model 
    def fit(self, train_test_split=0.75, number_of_days_to_test = None, number_of_days_to_train = None, n_dropout=0.2, should_tune_model=False):
        normalized_df, resampled_df = self.process(self.frequency, self.df, self.train_frequency, train_test_split, number_of_days_to_test, number_of_days_to_train)
        # split into train and test
        self.train, self.test, self.split_at = self.split_dataset(
            normalized_df.values, 
            len(resampled_df) / (24 * self.multiple),
            train_test_split,
            number_of_days_to_test,
            number_of_days_to_train
        )

        n_input = 24 * self.multiple  ## lag period is taken as 24 hours of data
        if should_tune_model:
            epochs_opt, batch_size_opt, neurons_per_layer_opt = self.model_tune(
                epochs=[30], 
                batch_size = [64], 
                neurons_per_layer = [50, 100, 200]
            )
        else:
            epochs_opt = 5
            batch_size_opt = 64
            neurons_per_layer_opt = 100

        model_run_count = 0
        count = 0
        #loop to ensure that model retrains in the case when error remains constant in the first few epochs
        while count == 0 and model_run_count <= 2:
            self.model, n_epochs = self.build_model(
                self.train,
                n_input,
                self.train_frequency,
                epochs_opt,
                batch_size_opt,
                neurons_per_layer_opt,
                n_dropout,
            )
            model_run_count += 1
            count += 1
            if n_epochs <= 2:
                count = 0


    #helper function for resampling, standardizing and normalizing the dataset
    def process(self, frequency, df, train_frequency,train_test_split,number_of_days_to_test,number_of_days_to_train):
        if train_frequency == "60min" and frequency == "15min":
            df_resample = df.resample('H').mean()
        elif train_frequency == "15min" and frequency == "60min":
            df_resample = df.resample("15min").interpolate("linear")
        
        if self.is_yearly_forecasts:
            df_new = df.copy()
            df_resample_new = df_resample.copy()

            df_new.reset_index(inplace=True)
            df_resample_new.reset_index(inplace=True)

            split_date = df_new.iloc[0].values[0]

            df_test = df_resample_new.loc[(df_resample_new['datetime'] >= split_date - relativedelta(months=3)) & (df_resample_new['datetime'] < split_date)]
            if ((split_date - relativedelta(months=3)) == df_resample_new.iloc[0].values[0]):
                df_train = df_resample_new.loc[df_resample_new['datetime'] >= split_date]
            else:
                df_train_split_1 = df_resample_new.loc[(df_resample_new['datetime'] < (split_date - relativedelta(months=3)))]
                df_train_split_2 = df_resample_new.loc[(df_resample_new['datetime'] >= split_date)]
                df_train = pd.concat([df_train_split_2, df_train_split_1], sort = False)

            df_resample = pd.concat([df_train, df_test], ignore_index=True, sort=False)
            df_resample.set_index('datetime', inplace=True)

        df_resample = self.add_timeseries_features(df_resample, len(df_resample), train_frequency)
        days = len(df_resample) / (24 * self.multiple)
        if number_of_days_to_test:
            train_test_split_days = days - number_of_days_to_test
        elif number_of_days_to_train:
            train_test_split_days = number_of_days_to_train
        else:
            train_test_split_days = int(train_test_split * days)
        split_at = int(train_test_split_days * 24 * self.multiple)
        x_train, x_test = df_resample[:split_at], df_resample[split_at:]
        if not self.scaler:
            self.scaler = MinMaxScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        x_scaled = np.concatenate((x_train_scaled, x_test_scaled), axis=0)
        df_normalized = pd.DataFrame(x_scaled, columns=df_resample.columns, index=df_resample.index)

        #scaler for load to be used for inverse transform on the predictions and test data later in evaluate
        x_train_load, x_test_load = df_resample.iloc[:split_at,0], df_resample.iloc[split_at:,0] 
        x_train_load = np.array(x_train_load).reshape(-1,1)
        x_test_load = np.array(x_test_load).reshape(-1,1)
        _ = self.scaler.fit_transform(x_train_load)
        _ = self.scaler.transform(x_test_load)

        return df_normalized, df_resample

    #forecast the next 'x' number of timesteps based on historical data where 'x' is the # of input timesteps 
    #generates a dictionary of forecasts for different std dev. for forecast uncertatinty analysis
    def predict(self, history, n_input):
        # flatten data
        data = np.array(history)
        if self.forecast_type=='every_hour':
            input_x = data[-n_input:, :]
            input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
        else:
            data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
            # retrieve last observations for input data
            input_x = data[-n_input:, :]
            # reshape into [1, n_input, n]
            input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))

        yhat_list = []
        for _ in range(10):
            yhat = self.model.predict(input_x, verbose=0)
            yhat_list.append(yhat)

        yhat = np.mean(yhat_list, axis=0)
        std_dev = np.std(yhat_list, axis=0)

        sigma_list = []

        std_dev_parameter_list = [
            -3,
            -2,
            -1,
            1,
            2,
            3,
        ]  # -3 to 3 std dev corresponds to 99.9% of data, -2 to 2 std dev corresponds to 95% of data and so on
        for param in std_dev_parameter_list:
            sigma = yhat + param * std_dev
            sigma_list.append(sigma[0])

        sigma_dict = dict(zip(std_dev_parameter_list, sigma_list))

        # we only want the vector load (or the required data to forecast)
        yhat = yhat[0]

        return yhat, sigma_dict

    #function to get metrics when fresh forecasts are generated every hour/15min
    def calculate_result_metrics_hourly(self, test_df, predictions_every_hour_df,test_set_dates):
        #generating 24 columns (if hourly) or 96 (if 15-min) for the test dataset for error calculation
        test_df_new = pd.concat([test_df]*24*self.multiple, axis=1)
        # test_df_new.to_csv('test_h_15min.csv')
        #shifting the columns of test dataset starting from column 1 to 23 (if hourly) or 95(if 15-min) up by a timestep for error calculation
        for i in range(1,24*self.multiple):
            test_df_new.iloc[:,i] = test_df_new.iloc[:,i].shift(-i)
        test_df_new.fillna(0, inplace=True)
        # test_df_new.to_csv('test_h_60min_shifted.csv')
        #error metric calculation by taking difference of actual and predicted load values
        error_difference_list = [(predictions_every_hour_df[i].values - test_df_new.iloc[:,i].values) for i in range(len(predictions_every_hour_df.columns))]
        error_df = pd.DataFrame(error_difference_list)
        error_df = error_df.T
        # error_df.to_csv('after transpose errors.csv')
        error_df['date'] = pd.Series(test_set_dates)
        error_df.set_index('date', inplace=True)
        #generating rmse score for every hour/15min and overall averaged rmse score
        rmse_list = [np.sqrt(np.mean(error_df.iloc[0:len(error_df[i])-i,i]**2)) for i in range(len(error_df.columns))]
        rmse_score = np.mean(rmse_list)
        
        return rmse_list, rmse_score
   
   
    #to evaluate the generated forecasts by comparing with the test dataset and calculating the error metrics
    def evaluate(self):
        n_input = 24 * self.multiple
        # print(self.test.shape[0], self.test.shape[1], self.test.shape[2])
        if self.forecast_type=='every_hour':
            self.train = self.train.reshape(self.train.shape[0]*self.train.shape[1], self.train.shape[2])
            self.test = self.test.reshape(self.test.shape[0] * self.test.shape[1], self.test.shape[2])
            history = [x for x in self.train]    
        else:
            # history is a list of daily data
            history = [x for x in self.train]

        # walk-forward validation over each day
        predictions = []

        import collections

        std_dev_dict = collections.defaultdict(list)
        for i in range(len(self.test)):
            # predict the day
            yhat_sequence, sigma_dict = self.predict(history, n_input)
            # store the predictions
            predictions.append(yhat_sequence)
            for key in sigma_dict.keys():
                std_dev_dict[key].append(sigma_dict[key].tolist())
            # get real observation and add to history for predicting the next hour
            if self.forecast_type == 'every_hour':
                history.append(self.test[i])
            else:
                history.append(self.test[i, :])

        # predictions array for all days
        predictions = np.array(predictions)
        predictions_reshaped = predictions.reshape(predictions.shape[0],predictions.shape[1]).reshape(-1,1)
        # print(predictions.shape)
        if self.forecast_type == 'every_hour':
            test = self.test[:,0] #only load values in first column are considered
        else:
            test = self.test[:,:,0]
            # print('new test shape is:',test.shape)
        test = test.reshape(-1,1)
        # revert normalization
        predictions_rescaled = self.scaler.inverse_transform(predictions_reshaped)
        test_rescaled = self.scaler.inverse_transform(test)

        for key in sigma_dict:
            std_dev_arr = np.asarray(std_dev_dict[key])
            std_dev_arr = std_dev_arr.reshape(std_dev_arr.shape[0],std_dev_arr.shape[1]).reshape(-1,1)
            std_dev_dict[key] = self.scaler.inverse_transform(std_dev_arr)

        if self.train_frequency == "60min" and self.frequency == "15min":
            df_resample = self.df.resample('H').mean()
        elif self.train_frequency == "15min" and self.frequency == "60min":
            df_resample = self.df.resample("15min").interpolate("linear")
        else:
            df_resample = self.df

        if self.is_yearly_forecasts:
            df_new = self.df.copy()
            df_resample_new = df_resample.copy()

            df_new.reset_index(inplace=True)
            df_resample_new.reset_index(inplace=True)

            split_date = df_new.iloc[0].values[0]

            df_test = df_resample_new.loc[(df_resample_new['datetime'] >= split_date - relativedelta(months=3)) & (df_resample_new['datetime'] < split_date)]
            if ((split_date - relativedelta(months=3)) == df_resample_new.iloc[0].values[0]):
                df_train = df_resample_new.loc[df_resample_new['datetime'] >= split_date]
            else:
                df_train_split_1 = df_resample_new.loc[(df_resample_new['datetime'] < (split_date - relativedelta(months=3)))]
                df_train_split_2 = df_resample_new.loc[(df_resample_new['datetime'] >= split_date)]
                df_train = pd.concat([df_train_split_2, df_train_split_1], sort = False)

            df_resample = pd.concat([df_train, df_test], ignore_index=True, sort=False)
            df_resample.set_index('datetime', inplace=True)

        test_set_dates = pd.date_range(df_resample.index[self.split_at], periods=len(test_rescaled), freq=self.train_frequency)

        if self.forecast_type =='every_hour':
            predictions_every_hour_arr = predictions_rescaled.reshape(predictions.shape[0],predictions.shape[1])
            predictions_every_hour_df = pd.DataFrame(predictions_every_hour_arr)
            predictions_every_hour_df['date'] = pd.Series(test_set_dates)
            predictions_every_hour_df.set_index('date', inplace=True)
            test_arr = test_rescaled
            test_df = pd.DataFrame(test_arr)
            scores, score = self.calculate_result_metrics_hourly(test_df, predictions_every_hour_df, test_set_dates)
            self.summarize_scores("lstm", score, scores)
        else:
            predictions_midnight_arr = predictions_rescaled.reshape(predictions.shape[0],predictions.shape[1])
            test_midnight_arr = test_rescaled.reshape(self.test.shape[0],self.test.shape[1])
            score, scores, score_mbe, scores_mbe = self.calculate_result_metrics(test_midnight_arr, predictions_midnight_arr)
            self.summarize_scores("lstm", score, scores)
            self.summarize_scores("mbe", score_mbe, scores_mbe)


        if self.forecast_type=='every_hour':
            forecasts = predictions_every_hour_df
        else:
            forecasts = predictions_rescaled.flatten().tolist()

        result = {
            "overall_score": score,
            "scores_by_hour": scores,
            # "mbe_score": score_mbe[0],
            # "mbe_scores_by_hour": scores_mbe,
            "actual_load": test_rescaled.flatten().tolist(),
            "forecasted": forecasts,
            "datetime":test_set_dates,
            "std": std_dev_dict
        }
        return result

    # Build and train the model
    def build_model(self, train, n_input, train_frequency, epochs=5, batch_size=128, neurons=100, dropout=0.0):
        # dropout = 0.2
        # prepare data
        train_x, train_y = self.to_supervised(train, n_input, train_frequency)
        # define parameters
        if train_frequency == "60min":
            verbose, epochs, batch_size = 2, epochs, batch_size
        else:
            verbose, epochs, batch_size = 2, 7, 64
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

        input_shape = (n_timesteps, n_features)
        inputs = keras.Input(shape=input_shape)
        x = LSTM(neurons, recurrent_dropout=dropout, activation="relu")(inputs, training=True)
        x = Dropout(dropout)(x, training=True)
        y = RepeatVector(n_outputs)(x)
        x = LSTM(neurons, dropout=dropout, return_sequences=True, activation="relu")(y, training=True)
        x = Dropout(dropout)(x, training=True)
        x = TimeDistributed(Dense(neurons, activation="relu"))(x)
        outputs = TimeDistributed(Dense(1, activation="relu"))(x)
        model = keras.Model(inputs, outputs)
        model.compile(loss="mse", optimizer="adam")
        model.summary()
        # fit network
        early_stopping = EarlyStopping(monitor="loss", min_delta=1e-6, mode="min", patience=3, verbose=verbose)
        history = model.fit(
            train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[early_stopping]
        )
        n_epochs = len(history.history["loss"])

        # model.save('rnn_model.h5')

        return model, n_epochs

    # convert history into inputs and outputs
    def to_supervised(self, train, n_input, train_frequency, n_out=24):
        if train_frequency == "60min":
            n_out = 24
        else:
            n_out = 24 * 4
        # flatten data
        data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
        X, y = [],[]
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for each instance
            if out_end < len(data):
                X.append(data[in_start:in_end, :])
                y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1
        return np.array(X), np.array(y)

    # add timeseries variables- day of the week dummy + holidays
    def add_timeseries_features(self, df, duration, train_frequency):
        if train_frequency == "60min":
            multiplier = 1
        else:
            multiplier = 4
        df["prev24HrAveLoad"] = float(0)
        x = [0] * 24 * multiplier
        a = df["load"].tolist()
        for i in range(len(a)):
            x.append(a[i])
        for i in range(duration):
            df["prev24HrAveLoad"][i] = np.mean(x[i + 1 : i + (24 * multiplier) + 1])

        df["prevDaySameHourLoad"] = df["load"].shift(24 * multiplier, fill_value=0)
        df["prevWeekSameHourLoad"] = df["load"].shift(24 * 7 * multiplier, fill_value=0)
        for hour in range(24):
            df["hour" + str(hour)] = 0
            df["hour" + str(hour)][df.index.hour == hour] = 1
        for day in range(7):
            df["day" + str(day)] = 0
            df["day" + str(day)][df.index.weekday == day] = 1
        startDate = df.index[0]
        endDate = df.index[-1]
        zipcode = 92104
        start = pd.Timestamp(startDate)
        end = pd.Timestamp(endDate)
        # df_temperature_from_api = esap_client.get_temperature_for_site(zipcode, start, end)
        # df_temperature = df_temperature_from_api[['observationTimeUtcIso','temperature']]
        # df_temperature.index = pd.to_datetime(df_temperature['observationTimeUtcIso'])
        # local_timezone = pytz.timezone("America/Los_Angeles")
        # df_temperature.index = df_temperature.index.tz_convert(local_timezone)
        # df_temperature.index = df_temperature.index.tz_localize(None)
        # df_temperature = df_temperature.resample('H').mean()
        # df_temperature = df_temperature.fillna(method='ffill')
        # add holiday feature
        cal = calendar()
        holidays = cal.holidays(start=df.index.min().date(), end=df.index.max().date())
        dates_list = []
        for day in holidays:
            dates = pd.date_range(start=day, periods=24*self.multiple, freq = '1H')
            dates_list.append(dates)

        flat_list = [item for sublist in dates_list for item in sublist]
        dates_arr = np.array(flat_list)
        df['holiday'] = df.index.isin(dates_arr)
        df['holiday'] = df['holiday'].astype(int)

        df1 = df[df['holiday']==1]


        # cal = calendar()
        # holidays = cal.holidays(start=self.df.index[0], end=self.df.index[-1])
        # df["holiday"] = [0 for x in range(df.shape[0])]
        # for holiday in holidays:
        # #     day = holiday.strftime("%Y-%m-%d")
        #     df[day]["holiday"] = True
        return df

    # evaluate one or more daily forecasts against expected values

    def calculate_result_metrics(self, actual, predicted):
        scores = []
        scores_mbe = []
        # calculate an RMSE score for each hour
        for i in range(actual.shape[1]):
            # calculate mse
            mse = mean_squared_error(actual[:, i], predicted[:, i])
            mbe = np.mean(predicted[:, i] - actual[:, i])
            # calculate rmse
            rmse = sqrt(mse)
            # store
            scores.append(rmse)
            scores_mbe.append(mbe)
        # calculate overall RMSE
        s = 0
        s_mbe = 0
        for row in range(actual.shape[0]):
            for col in range(actual.shape[1]):
                s += (actual[row, col] - predicted[row, col]) ** 2
                s_mbe += predicted[row, col] - actual[row, col]
        score = sqrt(s / (actual.shape[0] * actual.shape[1]))
        score_mbe = s_mbe / (actual.shape[0] * actual.shape[1])
        
        return score, scores, score_mbe, scores_mbe

    #tuning the hyperparameters of the model
    def hyperparam_tuning(self, epochs_list, batch_size_list, neurons_per_layer_list):
        print("-----TUNING THE MODEL PARAMETERS----")
        rmse_list = []
        ep_list = []
        batch_list = []
        npl_list = []
        n_input = 24 * self.multiple
        for epoch in epochs_list:
            for batch in batch_size_list:
                for neuron in neurons_per_layer_list:
                    self.model, n_epochs = self.build_model(
                        self.train, n_input, self.train_frequency, epoch, batch, neuron
                    )
                    results = self.evaluate()
                    rmse_list.append(results["overall_score"])
                    ep_list.append(epoch)
                    batch_list.append(batch)
                    npl_list.append(neuron)

        # Return hyperparameteres corresponding to lowest error metric
        results = pd.DataFrame(
            {"epochs": ep_list, "batch_size": batch_list, "neurons_per_layer": npl_list, "rmse": rmse_list}
        )
        temp = results[results["rmse"] == results["rmse"].min()]
        epochs_opt = temp["epochs"].values[0]
        batch_size_opt = temp["batch_size"].values[0]
        neurons_per_layer_opt = temp["neurons_per_layer"].values[0]
        # print(results.head(2))

        return epochs_opt, batch_size_opt, neurons_per_layer_opt, results

    #helper function to facilitate model tuning (tunes epochs, batch size and # of neurons per RNN layer)
    def model_tune(self, epochs=[30], batch_size = [64], neurons_per_layer = [50, 100, 200]):
        epochs_list = epochs
        batch_size_list = batch_size
        neurons_per_layer_list = neurons_per_layer

        epochs_opt, batch_size_opt, neurons_per_layer_opt, results = self.hyperparam_tuning(
            epochs_list, batch_size_list, neurons_per_layer_list
        )

        return epochs_opt, batch_size_opt, neurons_per_layer_opt
