import pandas as pd
import numpy as np
import os
import time, datetime
from datetime import datetime, timedelta
from google.cloud import storage
from google.oauth2 import service_account
from tensorflow import keras
from keras.models import load_model
import io
import h5py
import tensorflow
from tensorflow.python.lib.io import file_io   
import requests

from esap_forecast.models import RNN


PUT_REQUEST = "put"
GET_REQUEST = "get"
POST_REQUEST = "post"

ESAP_API_BASE = 'https://apigw.edf-esap.com/esap-api/v1/'
AUTH0_DOMAIN = os.environ.get('AUTH0_DOMAIN', 'edf-esap.auth0.com')
AUTH0_AUDIENCE = os.environ.get('AUTH0_AUDIENCE', 'https://edf-esap')
AUTH0_CLIENT_ID = os.environ.get('AUTH0_CLIENT_ID','L1OyNPrjJQbjOZU0XlkWdriNJd9MTBaV')
AUTH0_CLIENT_SECRET = os.environ.get('AUTH0_CLIENT_SECRET','-oHkQQrsiSBazLmThaGeuAXczR4-NRSzKOe5eF6t227M96Zdp5Gh0O9v6cYsE9um')

def get_auth_headers():
    url = f'https://{AUTH0_DOMAIN}/oauth/token'
    payload = f'audience={AUTH0_AUDIENCE}&grant_type=client_credentials&client_id={AUTH0_CLIENT_ID}&client_secret={AUTH0_CLIENT_SECRET}'
    headers = {
        'Content-Type': "application/x-www-form-urlencoded"
    }
    response = requests.request("POST", url, data=payload, headers=headers)
    token = response.json()['access_token']
    return token

def make_request(request_type, url, body={}, token=None, reattempt=False):

    auth_header = {'Authorization': f'Bearer {token}'}

    request_attempt = requests.request(
        method=request_type,
        url=url,
        json=body,
        headers=auth_header
    )

    if request_attempt.status_code < 400:
        return request_attempt
    elif not reattempt:
        token = get_auth_headers()
        return make_request(request_type, url, body, token, reattempt=True)
    else:
        if request_attempt.status_code == 401:
            raise Exception("Unable to make request, can't get a valid auth token")
        else:
            raise Exception(request_attempt.json())

def get_proposal_data(proposal_id, data_type):
    api_proposal_data = f'{ESAP_API_BASE}proposal/{proposal_id}/data/{data_type}'

    data_request = make_request(request_type=GET_REQUEST, url=api_proposal_data)

    return data_request

def get_proposal_from_esap(proposal_id, include_temp=False):
    proposal_request = get_proposal_data(proposal_id, 'load')
    data_json = proposal_request.json()
    df = pd.DataFrame(data_json.get('data'))
    df.index = pd.to_datetime(df['date'])
    df['load'] = pd.to_numeric(df['value'], errors = 'raise')
    df.drop(columns=['date', 'value'], inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns = {'date':'datetime'}, inplace=True)
    return df

def get_data(passed_df):
    """
    Args:
        passed_df (dataframe)
    Returns:
        [dataframe]: returns df - the cleaned dataframe
    """        
    df = passed_df.copy()
    value = passed_df['value']
    date = pd.to_datetime(df['datetime'])
    start = date.iloc[0]
    end = date.iloc[-1] + timedelta(minutes=15)  
    start_date = start.strftime("%m/%d/%Y")
    end_date = end.strftime("%m/%d/%Y")
    dates = pd.date_range(start=start_date,end=end_date, freq='15min')
    dates = dates[:-1]
    df = pd.DataFrame( {'load': value, 'datetime': dates })
    df = df.set_index('datetime')
    df = df.replace('?', np.nan)
    df = df.astype(np.float).fillna(method='bfill')
    return df

def upload_blob(gcs_creds, bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket """
    storage_client = storage.Client(credentials=gcs_creds, project=gcs_creds.project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, bucket_name
        )
    ) 


def main(proposal_id, model_file, test_days):
    df = get_proposal_from_esap(proposal_id)
    df.rename(columns = {'load':'value'}, inplace=True)
    df_new = get_data(df)
    my_model = RNN(df_new)
    my_model.fit(df_new, number_of_days_to_test=int(test_days))

    curr_path = os.path.abspath(os.getcwd())
    my_model.model.save(f'{curr_path}/{model_file}')

    # #     # FIXME This needs to be stored as a secret somehow!
    gcs_creds_json = {
        "type": "service_account",
        "project_id": "edfre-saf-esap-kf-c3ee7244",
        "private_key_id": "94433f7dbb811f0f5d2e82cc7377fc71444dbcf9",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7CZtHmcDxc9D9\nS8f54ml9mRa+G5swD/tZtZBXvK/1c65QpmWe52Ajc/MEVxA74ToNB9QiI4nNoQjM\nAmJNSkz+KuWi/HhLp12awvf6uCdU/odRM8jT/wt+2AgTuas6ML51D9fgszJ/+kWP\ntI2DgPaUquwCUPpF8dyiT4jmBMDTS5RdZ1nN9BcmEvuYgdMo7mQG8DP1JXXwxeo/\nqRVGP2z0tAUsm4GWhXY4/u/FKKmMY+DdC1nGquQCyLNPCOUXCogV0def0BLbB/Qp\nqIEMvHEYZF8MZ3peY3//30Artq5/hDHgACTpIMsxnM8soasYB5x4kKK/sU0Ev1zI\n+7vEj6m5AgMBAAECggEAQXTFtoOdDgMZryOPuyhdAbsLRgHUSDg3nzgW6VWcANr4\ntLnfE4Lm1tFzyV+My1/xmHDWcRId3mxOb1MgQutPUZ4Cmo0frl5GuGwmK8S54xlm\nkuj8DhESXVROU1TlkypO5Rnj03vzHu2f3YBzDAQch3/fs5nSVJslu1n5T4J3Vxpg\nObiAeVX8gWrfxErW4ckAdYhNSj4UTTXHtsPhK4QVM7FWAYrK2NwWgk45bku5Hw/J\n0GvP7fdA/sh2p8sO0DHNMXWErSZ3NspifLYGkE23Ptns/ANztC7NyOQ+pnfuds6+\nk9SjR1TYzzuliavFzOds8LELNGLJbphnKzAHDcrqJwKBgQD5f0NlW7GDCYdvOPS8\not8foOrybo0DNxdPITvS3eMqK5kOG7cNr1/Lthd4/aZzTehdZtU/zwJ3E8182O46\nWerwqr72wLM7LfxmIHMETKCMxmzGHaRfC9PFKKqNZYko9Rjp4ptfp8/IJb10LPUO\nx1UsSvAwa+Q4M6i+KVxEtKB/hwKBgQC/6ZcAliatDbCv3DFrl2lvPhHcCvJ+tbUi\n6uikV4aSlVTJSCrlTINezAFeI7sThyQLxDxtHRkQirXWcIx/rIkpEgdUf+miSsav\nDpEBQNhldIQ0IEmXkV/AYVlHzDRJUgnvdxq46/jwSzRkyiOcnVeWx6AaC1yM/D3R\nEtUIfCRcvwKBgDTJkk7dqZ8Z4wfLOyy2IRMmDs+gSEGH5GyfkXK585g2hTmQ75f7\npP8K1ciJkjAPKbypRzEq8VCUZgOmOjEqWST2W1UFzGYXArHw56TfOZDPYrBAEUjy\nzamHQx6LfwulX9IMWedRMAsewQjVgjvQPNsUN+Fm0nB5rZgeVCFNPi/9AoGAPRug\nHWuyPQBS9mxiwCOiSu20uiAyPu6VEt1B3rKQAzTRnpYAMqs+WX0UImm2M+2gKK7/\nnq7ZQE8qv3FvC2hg9FbtRpbESg0NXsiAm6mOPh/vyLlPNZfwqU1WJGp9a6tXxoxi\nUeexiLIpqL5EwKXJMhW2gRSFD89xz5TziJQe7/kCgYEA6kst3FbpoqrUf1Jwc84O\nL0ZvK2gF5SZQz/myAYrShyHtn9QW1+DjVK7vdrlAXO1SeWvL+z2P+ptdh1qPx7x4\nCyWeL+Kp51VMFA7UbGnkZ2AsmKi7C7yd8aK6M0SSX5Gb+eD62lI9kUcsV45DHtb7\nGKnI/saWXZogH6VFBqVS+RM=\n-----END PRIVATE KEY-----\n",
        "client_email": "rajat-worker-sa@edfre-saf-esap-kf-c3ee7244.iam.gserviceaccount.com",
        "client_id": "116667093576904413968",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/rajat-worker-sa%40edfre-saf-esap-kf-c3ee7244.iam.gserviceaccount.com"
    }

    # Turns JSON into acceptable authentication method for storage bucket
    gcs_creds = service_account.Credentials.from_service_account_info(gcs_creds_json)
    client = storage.Client(credentials=gcs_creds, project=gcs_creds.project_id)
    bucket = client.bucket('edf-rajat')
    upload_blob(gcs_creds, 'edf-rajat',f'{curr_path}/{model_file}', f'{model_file}' )


if __name__ == "__main__":
    main('482a1ffa-6dbf-4a51-84c3-b7267cfc65ff','sonoma_forecasts.h5', 2)