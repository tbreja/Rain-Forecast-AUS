# Import the Library and dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from scipy.stats import zscore
import itertools

class forecast_rain():
    def load_data(self, df):
        data = pd.read_csv(df, delimiter=',')
        self.df = data.copy()
        return data
    def nulled_value(self, data):
        missing = pd.DataFrame()
        missing['column'] = data.columns
        missing['value'] = data.isnull().sum().values
        fig = px.bar(
            data_frame=missing.sort_values('value'),
            x='value',
            y='column',
            orientation='h',
            title = 'Total Missing Value of each Columns',
            height=700,
            width=500)
        fig.show()
        return missing
    
    def clean(self, data):
        #for missing value in int or float type, we replace the nan value with mean value for each columns.
        for key, value in data[['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm']].iteritems():
            mean = data[key].mean()
            data[key].fillna(value=mean,inplace=True)
        #For missing value in object type, we drop the row that contain nan value.
        data.dropna(axis=0,inplace=True)
        data.describe()
        #make the date into days and months 
        data['Date'] = pd.to_datetime(data['Date'])
        data['Day'] = data['Date'].apply(lambda i: i.day)
        data['Month'] = data['Date'].apply(lambda i: i.month)
        data.drop(columns={'Date','RISK_MM'},inplace=True) 
        self.df_preproccessing = data.copy()
        return data     

    def remove_outliers(self, data):
        #Encode the labeled value into integer
        LE = LabelEncoder()
        for key, value in data[['WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow']].iteritems():
            data[key] = LE.fit_transform(data[key])
        z_scores = stats.zscore(data)
        abs_zscores = np.abs(z_scores)
        filtered = (abs_zscores < 3).all(axis=1)
        new_data = data[filtered]
        print('We have remove', str(data.shape[0] - new_data.shape[0]) + ' outliers')
        return new_data

    def modeling(self, data):
        X = ['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','WindGustDir','WindDir9am','WindDir3pm','RainToday']
        Y = ['RainTomorrow']
        x = np.array(data[X])
        x = StandardScaler().fit(x).transform(x)
        y = np.array(data[Y])
        
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state = 500)
        from sklearn.metrics import accuracy_score
        LR = LogisticRegression(C=0.05).fit(x_train,y_train)
        y_pred = LR.predict(x_test)
        self.y_real = np.copy(y_test)
        self.yhat = np.copy(y_pred)
        print(classification_report(y_test,y_pred))
        print('Your model have accuracy of : ' + str(100*accuracy_score(y_test, y_pred)) + ' %')
        

    def plot_confusion_matrix(self,cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        return self
