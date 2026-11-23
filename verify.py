"""Независимый пересчёт медианы по выгрузке и защита от разрывов данных."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from analyze import align, spans, topic
from pyulog import ULog

base=Path(__file__).resolve().parent
result=json.loads((base/'results/result.json').read_text(encoding='utf-8'))
source=json.loads((base/'source.json').read_text(encoding='utf-8'))
assert hashlib.sha256((base/'flight.ulg').read_bytes()).hexdigest()==source['sha256']==result['sha256']
with (base/'results/aligned.csv').open(encoding='utf-8',newline='') as f:
    selected=[r for r in csv.DictReader(f) if r['selected_window']=='1']
median=statistics.median(float(r['voltage_v']) for r in selected)
assert len(selected)==result['window_samples']==20
assert abs(median-result['median_voltage_v'])<1e-10
assert all(r['hover']=='1' for r in selected)
assert max(float(r['time_s']) for r in selected)-min(float(r['time_s']) for r in selected)>3.7
# Не подставлять будущую запись или устаревшее значение на разрыве.
values,ages=align(np.array([.5,1.,1.1,2.]),{'timestamp':np.array([1000000,3000000]),'x':np.array([10,30])},'x',.2)
assert np.isnan(values[0]) and values[1]==10 and values[2]==10 and np.isnan(values[3])
assert spans([True,True,True,True],np.array([0.,.2,3.,3.2]),.5)==[(0,1),(2,3)]
assert spans([True,False,True],np.array([0.,.2,.4]),.5)==[(0,0),(2,2)]
insufficient=json.loads((base/'results/target20_insufficient.json').read_text(encoding='utf-8'))
assert insufficient['status']=='insufficient_data' and insufficient['hover_duration_s']<20
# Поля и медиана отдельно подтверждаются прямо из ULog, без CSV и анализатора.
ulog=ULog(str(base/'flight.ulg'))
battery=next(d.data for d in ulog.data_list if d.name=='battery_status' and d.multi_id==0)
t=(battery['timestamp'].astype(float)-float(battery['timestamp'][0]))/1e6
v=[float(value) for time,value in zip(t,battery['voltage_v'])
   if result['window_start_s']-1e-8 <= time <= result['window_end_s']+1e-8]
assert len(v)==20 and abs(statistics.median(v)-median)<1e-10
report={'checks_passed':['source_hash','csv_median','raw_ulog_median','selected_hover','no_future_or_stale_alignment',
                         'gap_separates_segments','invalid_sample_separates_segments','insufficient_20_seconds'],
        'median_voltage_v':median,'window_samples':len(selected)}
(base/'results/verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('Проверки пройдены:',len(report['checks_passed']))
