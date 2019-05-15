import pandas as pd
import time


n_trips = 1000
def estimate(t, total_t, processed, left):
    """calculates the mean time it takes to process one chunk and reports the remaining time based on this estimation."""
    t_ = time.time()
    diff = t_ - t
    total_t += diff
    chunks_processed = processed // n_trips
    if not chunks_processed:
        return t_, total_t
    chunks_left = (left + n_trips - 1) // n_trips
    time_per_chunk = total_t / chunks_processed
    p = processed / (processed + left)
    print('\r{}% done in {} mins, remaining time: {} mins'.format(round(100 * p, 2), round(total_t / 60, 2), round((chunks_left * time_per_chunk) / 60, 2)), end = '')
    #print('\rTime: {}'.format(round(t_ - t, 2)), end = '')
    return t_, total_t


def groupby(df):
    """basically implements df.groupby('card_number') with additional time estimation."""
    t, total_t = time.time(), 0
    processed, curr = 0, n_trips
    
    for (key, value) in df.groupby('card_number'):
        processed += len(value)
        if processed >= curr:
            t, total_t = estimate(t, total_t, processed, len(df) - processed)
            curr += n_trips
        yield (key, value)
    t, total_t = estimate(t, total_t, len(df), 0)