'''

    Access the BigQuery EPA Air Quality dataset.

    Before this will run, you need an API key. Go to
    https://cloud.google.com/docs/authentication/getting-started
    and follow the steps there to get the key onto your computer.

'''

from bq_helper import BigQueryHelper
import pandas as pd

aq_data = BigQueryHelper('bigquery-public-data', 'epa_historical_air_quality')


# California
state_code = '06'

# Los Angeles
county_code = '067'
site_num = '1602'

date = '2018-08-11'

sql = "SELECT * FROM `bigquery-public-data.epa_historical_air_quality.air_quality_annual_summary` WHERE state_code = '16'"

tables = {
    'co': '0015',
    # 'hap',
    'no2': '0010',
    # 'nonoxnoy',
    # 'o3',
    'pm10': '2001',
    # 'pm25_frm',
    # 'pm25_nonfrm',
    # 'pm25_speciation',
    'pressure': '0015',
    'rh_and_dp': '0015',
    # 'so2',
    'temperature': '0015',
    'voc': '0015',
    'wind': '0015'
}

dfs = [ ]

for pollut, code in tables.items():
    sql = f"SELECT * FROM `bigquery-public-data.epa_historical_air_quality.{pollut}_hourly_summary` WHERE state_code = '{state_code}' AND county_code = '{county_code}' AND site_num = '{code}' AND date_local = '{date}'"

    dfs.append(aq_data.query_to_pandas(sql))











