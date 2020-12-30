#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import sys

def interpolate(df):
    new_index = pd.Index(range(df.index.max() + 1))
    interpolate_df = pd.DataFrame(index=new_index).join(df)
    for i, index in enumerate(df.index):
        try:
            index2 = df.index[i+1]
        except:
            break
        if isinstance(df.loc[index, 'x'], np.int64) and df.loc[index, 'x'] >= 0 and \
            isinstance(df.loc[index2, 'x'], np.int64) and df.loc[index2, 'x']:
            interpolate_df.loc[index:index2, 'x'] = np.linspace(df.loc[index, 'x'], df.loc[index2, 'x'],
                                                                num=1+index2-index, endpoint=True)
            interpolate_df.loc[index:index2, 'y'] = np.linspace(df.loc[index, 'y'], df.loc[index2, 'y'],
                                                                num=1+index2-index, endpoint=True)

    return interpolate_df.fillna(-1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('input-file', metavar='input')
    ap.add_argument('output-file', metavar='output')
    args = ap.parse_args(sys.argv[1:])


    df = pd.read_csv(args.input, index_col=0)

    interpolate(df).to_csv(args.output)