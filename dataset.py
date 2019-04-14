'''

    This script returns a pandas dataframe with the data we're using.

    Columns we want in the dataset:
        PM10
        NO2
        NO
        CO
        Ozone
        Temperature
        Wind Speed
        Wind Direction?
        Pressure
        Date
        AQI

    There are currently 345 days that we have all the data for in 2018.

'''

import numpy as np
import pandas as pd


AQI_table = {
    'O3 (8 hour)': {
        'Good': '0-54',
        'Moderate': '55-70',
        'Sensitive': '71-85',
        'Unhealthy': '86-105',
        'Very Unhealthy': '106-200',
        'Hazardous': '200-400'
    },
    'PM10': {
        'Good': '0-54',
        'Moderate': '55-154',
        'Sensitive': '155-254',
        'Unhealthy': '255-354',
        'Very Unhealthy': '355-424',
        'Hazardous': '425-604'
    },
    'CO': {
        'Good': '0.0-4.4',
        'Moderate': '4.5-9.4',
        'Sensitive': '9.5-12.4',
        'Unhealthy': '12.5-15.4',
        'Very Unhealthy': '15.5-30.4',
        'Hazardous': '30.5-50.4'
    },
    'NO2': {
        'Good': '0-53',
        'Moderate': '54-100',
        'Sensitive': '101-360',
        'Unhealthy': '361-649',
        'Very Unhealthy': '650-1249',
        'Hazardous': '1250-2049'
    },
    'Index': {
        'Good': '0-50',
        'Moderate': '51-100',
        'Sensitive': '101-150',
        'Unhealthy': '151-200',
        'Very Unhealthy': '201-300',
        'Hazardous': '301-500'
    }
}


def in_range(pollutant_aqi: dict, concentration: float) -> str:
    for label, crange in pollutant_aqi.items():
        nums = crange.split('-')

        if float(nums[0]) <= concentration and concentration <= float(nums[1]):
            return label

    raise RuntimeException('Could not find range :(')


def aqindex(indexrange: str, concenrange: str, concentration: float) -> int:
    Ilow, Ihigh = map(lambda x: int(x), indexrange.split('-'))
    Clow, Chigh = map(lambda x: float(x), concenrange.split('-'))

    return round(((Ihigh - Ilow)/(Chigh - Clow)) * (concentration - Clow) + Ilow)





first_parameters = [
    'PM10 - LC',
    'Ozone',
    'Nitrogen dioxide (NO2)',
    'PM10 - LC',
    'Wind Speed - Resultant',
    'Wind Direction - Resultant',
]

second_parameters = [
    'Carbon monoxide',
    'Outdoor Temperature',
]


def build_df(df1, df2, df3):

    days = [ ]

    for param in first_parameters:
        days.append(set(df1.loc[df1['Parameter Name'] == param]['Day In Year (Local)'].unique()))

    for param in second_parameters:
        days.append(set(df2.loc[df2['Parameter Name'] == param]['Day In Year (Local)'].unique()))


    intersection = days[0].intersection(*days[1:])

    # Ignore leap years.
    intersection.discard(366)

    print(f'Days in intersection: {len(intersection)}')

    new_df = pd.DataFrame(columns = [
        'Day Number',
        'Date',
        'PM10',
        'O3',
        'NO2',
        'CO',
        'PM10 AQI',
        'O3 AQI',
        'NO2 AQI',
        'CO AQI',
        'Wind Speed',
        'Wind Direction',
        'Temperature'
    ])

    # Start building the new dataframe.
    for day in intersection:

        # Dataframe of just this days measurements.
        measure = df1.loc[df1['Day In Year (Local)'] == day]

        pm10 = measure.loc[measure['Parameter Name'] == 'PM10 - LC']
        o3 = measure.loc[measure['Parameter Name'] == 'Ozone']
        no2 = measure.loc[measure['Parameter Name'] == 'Nitrogen dioxide (NO2)']
        windspeed = measure.loc[measure['Parameter Name'] == 'Wind Speed - Resultant']
        windir = measure.loc[measure['Parameter Name'] == 'Wind Direction - Resultant']

        # Second dataframe.
        measure = df2.loc[df2['Day In Year (Local)'] == day]

        co = measure.loc[measure['Parameter Name'] == 'Carbon monoxide']
        temp = measure.loc[measure['Parameter Name'] == 'Outdoor Temperature']

        new_df = new_df.append({
            'Day Number': day,
            'Date': pm10['Date (Local)'].values[0],
            'PM10': pm10['Arithmetic Mean'].values[0],
            'O3': o3['Arithmetic Mean'].values[0],
            'NO2': no2['Arithmetic Mean'].values[0],
            'CO': co['Arithmetic Mean'].values[0],
            'PM10 AQI': df3.loc[day - 1]['PM10'],
            'O3 AQI': df3.loc[day - 1]['Ozone'],
            'NO2 AQI': df3.loc[day - 1]['NO2'],
            'CO AQI': df3.loc[day - 1]['CO'],
            'Wind Speed': windspeed['Arithmetic Mean'].values[0],
            'Wind Direction': windir['Arithmetic Mean'].values[0],
            'Temperature': temp['Arithmetic Mean'].values[0]
        }, ignore_index = True)

    return new_df



def plots():
    from matplotlib import pyplot as plt
    import seaborn as sns
    import matplotlib as mpl
    mpl.style.use('seaborn')

    plt.plot(range(new_df['CO AQI'].values.shape[0]), new_df['CO AQI'].values,
            label = 'Reported AQI')

    line = [ ]
    for concentration in new_df['CO'].values:
        crange = in_range(AQI_table['CO'], concentration)
        line.append(aqindex(
            AQI_table['Index'][crange],
            AQI_table['CO'][crange],
            concentration))

    plt.plot(range(len(line)), np.array(line), label='Estimated AQI')
    plt.legend(loc='best')
    plt.xlabel('Day in Year (2018)')
    plt.ylabel('Carbon Monoxide AQI')
    plt.title('Reported CO AQI vs. Calculated CO AQI')
    plt.show()
    # print(np.mean(new_df['CO AQI'].values - np.array(line)))


def get_dfs(year: int) -> tuple:
    df1 = pd.read_csv(f'data/daily_06_067_0010_{year}.csv')
    df2 = pd.read_csv(f'data/daily_06_067_0015_{year}.csv')
    df3 = pd.read_csv(f'data/aqidaily{year}.csv')

    return df1, df2, df3


def build_network_data():
    '''Testing on 2017 CO for now.'''

    df2015 = build_df(*get_dfs(2015))
    df2016 = build_df(*get_dfs(2016))
    df2017 = build_df(*get_dfs(2017))
    df2018 = build_df(*get_dfs(2018))

    n_rows = df2015.shape[0] + df2016.shape[0] + df2018.shape[0] - 3
    X = np.zeros((n_rows, 7), dtype=float)
    y = np.zeros(n_rows, dtype=float)

    row = 0

    for df in [df2015, df2016, df2018]:
        print(row, df.shape[0] - 1, row + df.shape[0] - 1)
        X[row : row + df.shape[0] - 1, :-1] = df[['Day Number', 'CO', 'CO AQI', 'Wind Speed', 'Wind Direction', 'Temperature']].values[:-1, :]
        X[row : row + df.shape[0] - 1, -1] = np.squeeze(df[['Day Number']].values[1:])
        y[row : row + df.shape[0] - 1] = np.squeeze(df[['CO']].values[1:])

        row += df.shape[0] - 1

    Xtest = np.zeros((df2017.shape[0] - 1, 7), dtype=float)
    Xtest[:, :-1] = df2017[['Day Number', 'CO', 'CO AQI', 'Wind Speed', 'Wind Direction', 'Temperature']].values[:-1, :]
    Xtest[:, -1] = np.squeeze(df2017[['Day Number']].values[1:])

    ytest = np.zeros(df2017.shape[0] - 1, dtype=float)
    ytest[:] = np.squeeze(df2017[['CO']].values[1:])

    plot_correlation(df2017)

    return X, Xtest, y, ytest


def plot_correlation(df: pd.DataFrame) -> None:
    '''Do some plotting.'''

    from matplotlib import pyplot as plt
    import seaborn as sns
    import matplotlib as mpl
    mpl.style.use('seaborn')
    from mpl_toolkits.mplot3d import Axes3D

    from sklearn.linear_model import LinearRegression
    from sklearn.decomposition import PCA

    X = np.vstack((df['CO'].values[:-1], df['Wind Direction'].values[:-1])).T
    y = np.array(df['CO'].values[1:])

    # plt.subplot(1, 2, 1)
    # ws = df['Wind Speed'].values
    # plt.plot(range(ws.shape[0]), ws)

    # plt.subplot(1, 2, 2)
    # co = df['CO AQI'].values
    # plt.plot(range(co.shape[0]), co)
    # plt.show()

    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X)

    regress = LinearRegression().fit(X_pca, y)
    y_predicted = regress.predict(X_pca)

    plt.scatter(X_pca, y)
    plt.plot(X_pca, y_predicted, 'r')

    plt.xlabel('Reduced CO AQI and Wind Speed')
    plt.ylabel('Carbon Monoxide AQI')

    plt.title('PCA on CO AQI Predictions')
    plt.show()

    regress = LinearRegression().fit(X, y)
    y_predicted = regress.predict(X)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(
    #     df['CO'].values[:-1],
    #     df['Wind Speed'].values[:-1],
    #     df['CO'].values[1:]
    # )

    # # ax.plot(X[:, 0], X[:, 1], y_predicted, '.', color='r')

    # ax.set_xlabel('Old AQI')
    # ax.set_ylabel('Wind Speed')
    # ax.set_zlabel('New AQI')

    # plt.show()








if __name__ == '__main__':

    print('hello!')
    # df1 = pd.read_csv('data/daily_06_067_0010_2018.csv')
    # df2 = pd.read_csv('data/daily_06_067_0015_2018.csv')
    # df3 = pd.read_csv('data/aqidaily2018.csv')

    # df = build_df(df1, df2, df3)

    arrays = build_network_data()


























