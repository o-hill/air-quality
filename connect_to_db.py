'''

    Access the BigQuery EPA Air Quality dataset.

    Before this will run, you need an API key. Go to
    https://cloud.google.com/docs/authentication/getting-started
    and follow the steps there to get the key onto your computer.

'''

from bq_helper import BigQueryHelper
import pandas as pd

aq_data = BigQueryHelper('bigquery-public-data', 'epa_historical_air_quality')



sql = "SELECT * FROM `bigquery-public-data.epa_historical_air_quality.air_quality_annual_summary` WHERE state_code = '16'"

idaho_annual_summary = aq_data.query_to_pandas(sql)



