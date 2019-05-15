import pandas as pd
import numpy as np
from collections import defaultdict
from .groupby import groupby, estimate
import time


def transform(df, offset_day = 2, weeks = [1, 2]):
    """transforms raw correspondences to passenger profiles of the form (week, day, hour) -> number of trips."""
    def parse_date(date):
        #print(date)
        date, timestamp = date.split()
        day = (int(date.split('.')[0]) - offset_day)
        week, day = weeks[day // 7], days[day % 7]
        hour = int(timestamp.split(':')[0])
        return (week, day, hour)

    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']  
    hours = range(24)

    cls = [(x, y, z) for x in weeks for y in days for z in hours] 
    tp = pd.DataFrame(columns = cls)
    tp.columns = pd.MultiIndex.from_tuples(tp.columns, names = ['week', 'day', 'hour'])
    tp.index.name = 'card_number'

    for (key, value) in groupby(df):
        d = defaultdict(int)
        for t in value['start_time']:
            d[parse_date(t)] += 1
        s = pd.Series(d)
        s.name = value['card_number'].drop_duplicates().iloc[0]
        tp = tp.append(s)
    
    return tp.fillna(0).applymap(int)


def extract_patterns(df):
    """adds patterns and schedules to weekly view."""
    def is_workday(trips):
        return ['H', 'W'][trips > 0]
    
    def pattern(schedule):
        """produces pattern as the shortest period of the given schedule with no more than one error."""
        def subpattern(schedule):
            i = len(schedule)
            while i > 0 and schedule[i - 1] == 'H':
                i -= 1
            j = 0
            while j < len(schedule) and schedule[j] == 'H':
                j += 1
            schedule = schedule[j:]
            n = len(schedule)
            wd = 0
            twd = sum(map(lambda x : 1 if x == 'W' else 0, schedule))
    
            for p in range(n):
                if schedule[p] != 'H':
                    wd += 1
                    if p > 0:
                        continue
                sample = schedule[:p + 1] * ((n + p) // (p + 1))
                sample = sample[:n]
                if twd % wd == 0 and sample == schedule:
                    return schedule[:p + 1]
            return schedule

        var = [subpattern(schedule)]
        for p in range(len(schedule)):
            if schedule[p] == 'H':
                nschedule = ''.join(e if i != p else 'W' for i, e in enumerate(schedule))
                sc = subpattern(nschedule)
                if len(sc) == 1:
                    continue
                var.append(sc)
        return min(var, key = lambda x : len(x))

    df_ = df.sum(level = [0, 1], axis = 1)
    df['schedule'] = df_.apply(lambda x : ''.join(is_workday(y) for y in x), axis = 1)
    df['pattern'] = df['schedule'].map(pattern)
    return df


def __compress(df):
    pattern = list(df.iloc[0])[-1]
    n_days = len(pattern)
    
    card_numbers, schedules = list(df.index), list(df['schedule'])
    df = df[df.columns.values[:-2]]
    
    cls = [(day + 1, hour) for day in range(n_days) for hour in range(24)]
    tp = pd.DataFrame(columns = cls)
    tp.columns = pd.MultiIndex.from_tuples(tp.columns, names = ['day', 'hour'])
    tp.index.name = 'card_number'
    
    pos = lambda x, offset : (x - offset + n_days) % n_days
    
    t, total_t = time.time(), 0
    for i, arr in enumerate(df.values):    
        if i:
            t, total_t = estimate(t, total_t, i, len(df) - i)
        offset = schedules[i].find('W')
        d = np.zeros((n_days, 24))
        for j, ax in enumerate(np.split(arr, len(arr) // 24)):
            d[pos(j, offset)] += ax
        s = pd.Series({(x + 1, y) : d[x][y] for x in range(n_days) for y in range(24)})
        s.name = card_numbers[i]
        tp = tp.append(s)
    
    tp = tp.applymap(int)
    tp['pattern'] = [pattern] * len(tp)
    return tp


def compress(df):
    """transforms a data frame from weekly to pattern-specific view, aggregating trips belonging to the same days according to the pattern."""
    mp = {}
    for (key, profile) in df.groupby('pattern'):
        mp[key] = __compress(profile)
    return mp


def filter_by_pattern_len(df, pattern_len = 7):
    """excludes all passengers with pattern lengths greater than the specified length."""
    dx = df.loc[:, 'pattern'].to_frame('pattern')
    dx = dx[dx.apply(lambda x : len(x.pattern) <= pattern_len, axis = 1)]
    return df.join(dx.drop(columns = 'pattern'), how = 'inner')