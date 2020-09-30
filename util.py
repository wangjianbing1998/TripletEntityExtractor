def get_schemas():
    import json

    from collections import defaultdict

    ds = defaultdict(list)
    with open('./raw_data/all_11_schemas', 'r', encoding='utf-8') as f:
        count_numbers = 0
        while True:
            line = f.readline()
            if line:
                count_numbers += 1
                spo = json.loads(line)
                ds[spo['predicate']].append((spo['object_type'], spo['subject_type']))
            else:
                break
    return ds