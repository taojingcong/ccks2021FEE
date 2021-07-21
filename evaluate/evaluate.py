import json

result_type = ["reason_region", "reason_product", "reason_industry", "result_region",
               "result_product", "result_industry"]
ans_path = '../data/dev.json'
dev_path = '../argument/2021-07-17_18-36-15/finnalAns.json'
pred = 0
right = 0
total = 0
id_idx = {}
f = open(ans_path, encoding="utf-8")
ans = [json.loads(line.strip()) for line in f]
for i in range(len(ans)):
    id_idx[ans[i]['text_id']] = i

g = open(dev_path, encoding="utf-8")
dev = [json.loads(line.strip()) for line in g]
for dev_line in dev:
    id = dev_line['text_id']
    if id in id_idx.keys():
        ans_line = ans[id_idx[id]]
        d = {}
        a = {}
        for r in dev_line['result']:
            rt = r['result_type'] + '#' + r['reason_type']
            if rt not in d.keys():
                d[rt] = {}
                for t in result_type:
                    d[rt][t] = set()
            for t in result_type:
                for it in r[t].split(','):
                    d[rt][t].add(it)

        for r in ans_line['result']:
            rt = r['result_type'] + '#' + r['reason_type']
            if rt not in a.keys():
                a[rt] = {}
                for t in result_type:
                    a[rt][t] = set()
            for t in result_type:
                for it in r[t].split(','):
                    a[rt][t].add(it)

        for rt in d.keys():
            if rt in a.keys():
                for t in result_type:
                    tmp = a[rt][t] & d[rt][t]
                    right += len(tmp)
                    total += len(a[rt][t])
                    pred += len(d[rt][t])

            else:
                for t in result_type:
                    pred += len(d[rt][t])

    else:
        for r in dev_line['result']:
            for t in result_type:
                pred += len(r[t].split(','))

print(right, pred, right)

p1 = right / pred
r1 = right / total
f1 = 2.0 * p1 * r1 / (p1 + r1)
log = f'p: {p1:.6f}, r: {r1:.6f}, f1: {f1:.6f}'
print(log)
