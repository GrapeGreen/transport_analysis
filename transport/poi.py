import pandas as pd
from .groupby import groupby
from collections import defaultdict
from sklearn.neighbors import DistanceMetric
from numpy import radians


dist = DistanceMetric.get_metric('haversine')
def haversine_distance(crd):
    return dist.pairwise(crd)[0][1] * 6371


def transfer_distance(a, b):
    """returns distance between locations in km."""
    coords = [[a.latitude_start, a.longitude_start], [b.latitude_end, b.longitude_end]]
    coords = [[radians(x) for x in y] for y in coords]
    return haversine_distance(coords)


def home(corr, distance = 0.5, threshold = 5):
    """determines home location of a passenger."""
    home = defaultdict(int)
    for (key, value) in corr.groupby('ddate'):
        value = value.drop(columns = ['corr_number']).reset_index().drop(columns = 'index')
        fj, lj = value.iloc[0], value.iloc[-1]
        if len(value) == 1:
            continue
        if transfer_distance(fj, lj) <= distance:
            home[fj['cluster_start']] += 1
            if fj['cluster_start'] != lj['cluster_end']:
                home[lj['cluster_end']] += 1
                    
    cand = [a for (a, b) in home.items() if b >= threshold]
    if not cand:
        return None
    return cand[0]
      
    
def work(corr, home, distance = 0.5, threshold = 5):
    """determines work location of a passenger."""
    def time_difference(a, b):
        def to_minutes(time):
            hours, minutes, seconds = map(int, time.split(':'))
            return minutes + 60 * hours # ignoring seconds, they are always equal to 00
        
        time_a, time_b = map(lambda x : x.split()[-1], [a.end_time, b.start_time])
        return to_minutes(time_b) - to_minutes(time_a)

    work = defaultdict(int)
    total = 0
    for (key, value) in corr.groupby('ddate'):
        value = value.drop(columns = ['corr_number']).reset_index().drop(columns = 'index')
        if len(value) == 1:
            continue
        for (i, journey_a), (j, journey_b) in zip(list(value.iterrows())[:-1], list(value.iterrows())[1:]):
            if time_difference(journey_a, journey_b) >= 360 and transfer_distance(journey_b, journey_a) <= distance:
                location_a, location_b = journey_a.cluster_end, journey_b.cluster_start
                if location_a != home:
                    work[location_a] += 1
                if location_b != location_a and location_b != home:
                    work[location_b] += 1
                    
    cand = [a for (a, b) in work.items() if b >= threshold]
    if not cand:
        return None
    return cand[0]


def get_poi(df, distance = 0.5, threshold = 5):
    """calculates points of interest (namely work and home) for passengers."""
    residence = pd.DataFrame(columns = ['card_number', 'work', 'home'])

    for (key, value) in groupby(df):
        home_location = home(value, distance, threshold)
        if home_location is None:
            continue
        work_location = work(value, home_location, distance, threshold)
        if work_location is None:
            continue
        if home_location == work_location:
            continue
        residence = residence.append({'card_number' : key, 'home' : home_location, 'work' : work_location}, ignore_index = True)
    residence['card_number'] = residence['card_number'].astype('int64')    
    return residence


def __filter_workdays(corr, stops = None):
    """eliminates all days on which the particular passenger didn't visit his work location."""
    if corr.empty:
        return corr
    
    dates = set()
    work = corr.iloc[0].work
        
    if stops:
        work_crd = [stops.loc[work].latitude, stops.loc[work].longitude]
    
    for (key, value) in corr.groupby('ddate'):
        work_vis = False
        for (id, row) in value.iterrows():
            # this is a bit weird, though it just performs a check whether one of the visited stops is in close proximity to work location.
            if stops and min(
                haversine_distance([
                    [row.latitude_start, row.longitude_start],
                    work_crd
                ]),
                haversine_distance([
                    [row.latitude_end, row.longitude_start],
                    work_crd
                ])
            ) <= 0.5:
                work_vis = True
                break
            if row.work in [row.cluster_start, row.cluster_end]:
                work_vis = True
                break
                
        if work_vis:
            dates.add(key)
            
    corr = corr[corr.ddate.isin(dates)]
    return corr


def filter_workdays(df):
    """eliminates all days on which passengers didn't visit their work locations. (wrapper for __filter_workdays)"""
    if not len(df):
        return df
    return pd.concat([__filter_workdays(value) for (key, value) in groupby(df)])