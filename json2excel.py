import os

path = 'output/final_text_spo_list_result/keep_empty_spo_list_subject_predicate_object_predict_output.json'
import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 6)
pd.set_option('precision', 2)

import json
from collections import defaultdict

ds = defaultdict(list)
with open(path, 'r', encoding='utf-8') as f:
    count_numbers = 0
    while True:
        line = f.readline()
        if line:
            count_numbers += 1
            r = json.loads(line)
            for spo in r['spo_list']:
                ds['object_type'].append(spo['object_type'])
                ds['predicate'].append(spo['predicate'])
                ds['object'].append(spo['object'])
                ds['subject_type'].append(spo['subject_type'])
                ds['subject'].append(spo['subject'])
                ds['text'].append(r['text'])
        else:
            break

pd.DataFrame(ds).to_excel(os.path.join(os.path.dirname(path), 'output.xlsx'))
