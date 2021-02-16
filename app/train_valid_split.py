usage = """
Usage:
    app/train_valid_split.py init
    app/train_valid_split.py --random-state=<int>
"""

import pandas as pd
from docopt import docopt
from sklearn.model_selection import train_test_split

args = docopt(usage)
df = pd.read_csv('bert_train.csv', index_col=0)

try:
    random_state = int(args['--random-state'])
except:
    random_state = None

train, valid = train_test_split(df, test_size=0.25, random_state=random_state, stratify=df.information_label, shuffle=True)

print(train.label.value_counts())
print(valid.label.value_counts())

train.to_csv('bert_train_split.csv')
valid.to_csv('bert_valid_split.csv')