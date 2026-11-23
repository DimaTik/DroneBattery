"""Один воспроизводимый расчёт по открытому PX4 ULog; не диагностика батареи."""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from pyulog import ULog


DEFAULTS = {
    'horizontal_speed_m_s': 0.3, 'vertical_speed_m_s': 0.2,
    'thrust_window_s': 2.0, 'thrust_peak_to_peak': 0.1,
    'thrust_min': 0.1, 'thrust_max': 0.9,
    'position_max_age_s': 0.25, 'land_max_age_s': 1.5,
    'thrust_max_age_s': 0.1, 'battery_max_gap_s': 0.5,
    'segment_min_s': 5.0, 'target_hover_s': 5.0, 'window_half_s': 2.0,
}


def topic(ulog, name):
    matches = [x.data for x in ulog.data_list if x.name == name and x.multi_id == 0]
    if len(matches) != 1:
        raise ValueError('Нет однозначной темы ' + name)
    data = matches[0]
    if np.any(np.diff(data['timestamp'].astype(np.int64)) <= 0):
        raise ValueError('Немонотонное время: ' + name)
    return data


def align(times, data, field, max_age):
    source_t = data['timestamp'].astype(float) / 1e6
    idx = np.searchsorted(source_t, times, side='right') - 1
    safe = np.maximum(idx, 0)
    age = times - source_t[safe]
    valid = (idx >= 0) & (age >= 0) & (age <= max_age)
    return np.where(valid, data[field][safe].astype(float), np.nan), age


def spans(mask, times, max_gap):
    groups, start = [], None
    for i, valid in enumerate(mask):
        if start is not None and (not valid or times[i] - times[i-1] > max_gap):
            groups.append((start, i-1)); start = None
        if valid and start is None:
            start = i
    if start is not None:
        groups.append((start, len(mask)-1))
    return groups


def analyze(path, output, config=None):
    started = time.perf_counter()
    config = {**DEFAULTS, **(config or {})}
    output.mkdir(parents=True, exist_ok=True)
    ulog = ULog(str(path))
    inventory = [{'topic': d.name, 'instance': d.multi_id, 'rows': len(d.data['timestamp']),
                  'fields': list(d.data)} for d in ulog.data_list]
    (output/'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    result = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'parameters': config,
              'status': 'insufficient_data', 'log_duration_s': (ulog.last_timestamp-ulog.start_timestamp)/1e6,
              'dropouts': len(ulog.dropouts)}
    try:
        battery = topic(ulog, 'battery_status')
        position = topic(ulog, 'vehicle_local_position')
        land = topic(ulog, 'vehicle_land_detected')
        thrust = topic(ulog, 'vehicle_thrust_setpoint')
        times = battery['timestamp'].astype(float)/1e6
        voltage = battery['voltage_v'].astype(float)
        vx, pos_age = align(times, position, 'vx', config['position_max_age_s'])
        vy, _ = align(times, position, 'vy', config['position_max_age_s'])
        vz, _ = align(times, position, 'vz', config['position_max_age_s'])
        vxy_valid, _ = align(times, position, 'v_xy_valid', config['position_max_age_s'])
        vz_valid, _ = align(times, position, 'v_z_valid', config['position_max_age_s'])
        landed, land_age = align(times, land, 'landed', config['land_max_age_s'])
        contact, _ = align(times, land, 'ground_contact', config['land_max_age_s'])
        maybe, _ = align(times, land, 'maybe_landed', config['land_max_age_s'])
        thrust = dict(thrust)
        thrust['magnitude'] = np.sqrt(sum(thrust[f'xyz[{i}]'].astype(float)**2 for i in range(3)))
        command, thrust_age = align(times, thrust, 'magnitude', config['thrust_max_age_s'])
        thrust_times = thrust['timestamp'].astype(float)/1e6
        spread = np.full(len(times), np.nan)
        for i, t in enumerate(times):
            lo = np.searchsorted(thrust_times, t-config['thrust_window_s'], side='left')
            hi = np.searchsorted(thrust_times, t, side='right')
            window = thrust_times[lo:hi]
            if len(window) < 2 or window[0] > t-config['thrust_window_s']+config['thrust_max_age_s']:
                continue
            if t-window[-1] > config['thrust_max_age_s'] or np.max(np.diff(window)) > config['thrust_max_age_s']:
                continue
            spread[i] = np.ptp(thrust['magnitude'][lo:hi])
        speed = np.hypot(vx, vy)
        valid = (np.isfinite(voltage) & (voltage > 0) & (battery['connected'] == 1)
                 & (vxy_valid == 1) & (vz_valid == 1) & (landed == 0) & (contact == 0) & (maybe == 0)
                 & (speed <= config['horizontal_speed_m_s']) & (np.abs(vz) <= config['vertical_speed_m_s'])
                 & (command >= config['thrust_min']) & (command <= config['thrust_max'])
                 & (spread <= config['thrust_peak_to_peak']))
        groups = [(a,b) for a,b in spans(valid,times,config['battery_max_gap_s'])
                  if times[b]-times[a] >= config['segment_min_s']]
        mask = np.zeros(len(times), dtype=bool)
        cumulative = np.zeros(len(times)); segment = np.full(len(times), -1)
        total = 0.0
        intervals=[]
        for n,(a,b) in enumerate(groups):
            mask[a:b+1] = True; segment[a:b+1] = n
            cumulative[a:b+1] = total + times[a:b+1]-times[a]
            duration=times[b]-times[a];total+=duration
            intervals.append({'start_s':float(times[a]-times[0]),'end_s':float(times[b]-times[0]),
                              'duration_s':float(duration),'samples':b-a+1})
        result.update(battery_samples=len(times), hover_duration_s=float(total), intervals=intervals)
        selected=np.zeros(len(times),dtype=bool)
        candidates=np.flatnonzero(mask & (cumulative>=config['target_hover_s']))
        if not len(candidates):
            result['reason']='Не накоплено заданное время пригодного висения'
        else:
            index=int(candidates[0]);target=times[index]
            selected=mask & (segment==segment[index]) & (np.abs(times-target)<=config['window_half_s'])
            samples=np.flatnonzero(selected)
            if len(samples)<5 or times[samples[-1]]-times[samples[0]] < config['window_half_s']:
                result['reason']='Недостаточное окно напряжения около выбранного момента'
            else:
                cells=np.unique(battery['cell_count'][selected])
                cell_count=int(cells[0]) if len(cells)==1 and cells[0]>0 else None
                declared=ulog.initial_parameters.get('BAT1_N_CELLS')
                if declared and cell_count!=int(declared):cell_count=None
                median=float(np.median(voltage[selected]))
                result.update(status='calculated', target_time_s=float(target-times[0]),
                    actual_hover_at_target_s=float(cumulative[index]), window_start_s=float(times[samples[0]]-times[0]),
                    window_end_s=float(times[samples[-1]]-times[0]), window_samples=len(samples),
                    median_voltage_v=median, cell_count=cell_count, parameter_bat1_n_cells=declared,
                    median_voltage_per_cell_v=median/cell_count if cell_count else None)
        columns={'time_s':times-times[0],'voltage_v':voltage,'horizontal_m_s':speed,'vertical_m_s':vz,
                 'thrust_command':command,'thrust_range_2s':spread,'position_age_s':pos_age,
                 'land_age_s':land_age,'thrust_age_s':thrust_age,'landed':landed,
                 'ground_contact':contact,'maybe_landed':maybe,'velocity_xy_valid':vxy_valid,
                 'velocity_z_valid':vz_valid,'hover':mask.astype(int),'segment':segment,
                 'accumulated_hover_s':cumulative,'selected_window':selected.astype(int)}
        with (output/'aligned.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(columns)
            writer.writerows(zip(*columns.values()))
        os.environ.setdefault('MPLCONFIGDIR',str(output/'.matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
        t=times-times[0]
        axes[0].plot(t,voltage,color='#265d88',linewidth=1.2);axes[0].set_ylabel('Напряжение, В')
        axes[1].plot(t,speed,label='Горизонтальная',linewidth=1)
        axes[1].plot(t,np.abs(vz),label='Вертикальная, модуль',linewidth=1)
        axes[1].set_ylabel('Скорость, м/с');axes[1].legend(loc='upper right')
        axes[2].plot(t,command,color='#694f80',linewidth=1);axes[2].set_ylabel('Команда тяги')
        axes[2].set_xlabel('Время от первой записи батареи, с')
        for ax in axes:
            ax.spines[['top','right']].set_visible(False)
            for n,(a,b) in enumerate(groups):ax.axvspan(t[a],t[b],color='#4b9364',alpha=.18,label='Пригодное висение' if n==0 else None)
            if selected.any():ax.axvspan(t[selected][0],t[selected][-1],color='#d09026',alpha=.45)
        from matplotlib.patches import Patch
        axes[0].legend(handles=[Patch(color='#4b9364',alpha=.18,label='Пригодное висение'),
                               Patch(color='#d09026',alpha=.45,label='Окно расчёта')],loc='upper right')
        fig.suptitle('Один открытый полёт PX4: отбор участков и расчёт напряжения')
        fig.savefig(output/'calculation.png',dpi=150);plt.close(fig)
    except (KeyError,ValueError) as error:
        result['reason']=str(error)
    result['calculation_seconds'] = round(time.perf_counter()-started, 6)
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=Path(__file__).with_name('flight.ulg'))
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('results'))
    parser.add_argument('--target-hover-seconds',type=float,default=DEFAULTS['target_hover_s'])
    args=parser.parse_args()
    print(json.dumps(analyze(args.input,args.output,{'target_hover_s':args.target_hover_seconds}),ensure_ascii=False,indent=2))
