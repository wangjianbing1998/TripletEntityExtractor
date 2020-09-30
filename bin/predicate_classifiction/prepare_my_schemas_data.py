import json

from collections import defaultdict

ds = set()
with open('../../my_data/train_data_my.json', 'r', encoding='utf-8') as f:
    count_numbers = 0
    while True:
        line = f.readline()
        if line:
            count_numbers += 1
            r = json.loads(line)
            for spo in r['spo_list']:
                ds.add((spo['predicate'], spo['subject_type'], spo['object_type']))
        else:
            break

ds = list(ds)

with open('../../my_data/all_11_schemas', 'w',encoding='utf-8') as f:
    json_data = defaultdict()
    for pre, sub, obj in ds:
        json_data['predicate'] = pre
        json_data['subject_type'] = sub
        json_data['object_type'] = obj
        j = json.dumps(json_data,ensure_ascii=False)
        f.writelines(j+'\n')

