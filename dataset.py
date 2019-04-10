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

    return ((Ihigh - Ilow)/(Chigh - Clow)) * (concentration - Clow) + Ilow




df1 = pd.read_csv('daily_06_067_0010_2018.csv')
df2 = pd.read_csv('daily_06_067_0015_2018.csv')
df3 = pd.read_csv('aqidaily.csv')


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

days = [ ]

for param in first_parameters:
    days.append(set(df1.loc[df1['Parameter Name'] == param]['Day In Year (Local)'].unique()))

for param in second_parameters:
    days.append(set(df2.loc[df2['Parameter Name'] == param]['Day In Year (Local)'].unique()))


intersection = days[0].intersection(*days[1:])

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

    from ipdb import set_trace as debug; debug()

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











