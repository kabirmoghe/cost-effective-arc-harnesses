#!/bin/bash
cd /Users/kabirmoghe/Developer/Thesis/src/metrics
for f in *.json; do
    [ -f "$f" ] || continue
    python3 -c "
import json
d = json.load(open('$f'))
keys = ['split','model','approach','baseline','total','correct','accuracy','token_usage']
print('$f:', {k: d[k] for k in keys if k in d})
" 2>/dev/null
done
