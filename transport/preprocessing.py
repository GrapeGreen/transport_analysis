import pandas as pd
from .groupby import groupby


def read_corr(path = None, nrows = None, sep = ',', card_type = None, stops = None, df = None):
    if df is None:
        df = pd.read_csv(path, nrows = nrows, sep = sep)
        df = df.rename(columns = {x : x.lower() for x in df})

    if card_type is not None:
        df = df[df.card_type_id == card_type]

    #print(len(df))
      
    irrelevant = set(['corr_type', 'start_group_id', 'end_group_id']) & set(df.columns.values)
    df = df.drop(columns = list(irrelevant))
    
    stop_info = ['id_stop', 'cluster', 'stop_name', 'longitude', 'latitude']
    if stops is None:
        stops = pd.read_csv('stops.csv')
        stops = stops[stop_info]
    
    for column in ['start', 'end']:
        df = pd.merge(df, stops, left_on = '{}_stop_id'.format(column), right_on = 'id_stop', how = 'inner').drop(columns = '{}_stop_id'.format(column))
        df = df.rename(columns = {x : '{}_{}'.format(x, column) for x in stop_info})
    
    df = df.sort_values(['card_number', 'corr_number']).reset_index(drop = True)
    return df


def visit_threshold(df, threshold = 7):
    return df.groupby('card_number').filter(lambda group : len(group) >= 2 * threshold and len(group.ddate.drop_duplicates()) >= threshold)
