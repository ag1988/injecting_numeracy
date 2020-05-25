
import json
import pandas as pd


def uppercase_first_char(string):
    return ''.join([c if i > 0 else c.upper() for i, c in enumerate(string)])


def drop_duplicates(df, subset, dict_cols, print_duplicates=False):
    for col in dict_cols:
        df[col] = df[col].apply(json.dumps)

    if print_duplicates:
        print("\nremoving duplicates:")
        duplicate_groups = [g for _, g in df.groupby(subset) if len(g) > 1]
        if duplicate_groups:
            print(pd.concat(duplicate_groups))
        else:
            print("[]")

    df.drop_duplicates(subset=subset, inplace=True)

    for col in dict_cols:
        df[col] = df[col].apply(json.loads)

