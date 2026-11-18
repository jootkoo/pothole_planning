"""Benchmark occupancy-grid A*, RRT, and A*+DWA with separately reported timing.

Run from any directory: python Benchmark/benchmark.py --trials 10
Timing is taken from planner internals to avoid double-counting geometry validation.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import math
from pathlib import Path
import statistics
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
for folder in (ROOT, ROOT/'Benchmark', ROOT/'AStar', ROOT/'RRT', ROOT/'DWA'):
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))
from common import scenario, available_scenarios
from geometry_validation import validate_path
from metrics import calculate_metrics, path_length

MODULE_PATHS = {'AStar':'AStar/astar.py','RRT':'RRT/rrt.py','DWA':'DWA/dwa.py'}
FIELDS = ['algorithm','scenario','scenario_name','trial','seed','success','failure_reason',
          'grid_build_ms','search_ms','validation_ms','warm_total_ms','cold_total_ms',
          'time_to_valid_path_ms','time_to_valid_path_warm_ms','metric_computation_ms',
          'effort','effort_unit','validation_checks','validated_candidates','rejected_candidates',
          'path_length_m','min_clearance_m','total_turn_rad','max_turn_rad','waypoints',
          'reference_path_length_m','error']

def load_module(name, relative):
    path=ROOT/relative
    spec=importlib.util.spec_from_file_location('benchmark_'+name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

def run_trial(algorithm,module,sid,trial,seed,iterations,dwa_steps,resolution,field_resolution):
    row={k:None for k in FIELDS}
    row.update(algorithm=algorithm,scenario=sid,trial=trial,seed=seed if algorithm=='RRT' else None,
               success=0, failure_reason='',error='',validation_checks=0,validated_candidates=0,rejected_candidates=0)
    try:
        road,start,goal,name=scenario(sid)
        row['scenario_name']=name
        if algorithm=='AStar':
            planner=module.AStarPlanner(road,resolution=resolution)
            path,_,effort=planner.plan(start,goal)
            row['effort_unit']='expanded_nodes'
        elif algorithm=='RRT':
            planner=module.RRTPlanner(road,seed=seed,max_iterations=iterations,resolution=resolution)
            path,_,effort=planner.plan(start,goal)
            row['effort_unit']='iterations'
        else:
            planner=module.DWAPlanner(road,planning_resolution=resolution,field_resolution=field_resolution)
            path,reference,_,effort=planner.plan(start,goal,max_steps=dwa_steps)
            row['effort_unit']='control_steps'
            row['reference_path_length_m']=path_length(reference)
        row.update(effort=effort,grid_build_ms=planner.grid_build_ms,search_ms=planner.search_ms,
                   validation_ms=planner.validation_ms,validation_checks=getattr(planner,'validation_checks',0),
                   validated_candidates=getattr(planner,'validated_candidates',int(path is not None)),
                   rejected_candidates=getattr(planner,'rejected_exact',0))
        row['warm_total_ms']=row['search_ms']+row['validation_ms']
        row['cold_total_ms']=row['grid_build_ms']+row['warm_total_ms']
        # The planner has already performed original-geometry validation. Do not revalidate
        # inside the timing window. Endpoint checks are common and cheap, and failures are explicit.
        endpoints_ok=bool(path) and math.dist(path[0][:2],start)<=0.70 and math.dist(path[-1][:2],goal)<=0.70
        if path and endpoints_ok:
            row['success']=1
            t=time.perf_counter()
            row.update(calculate_metrics(path,road))
            row['metric_computation_ms']=(time.perf_counter()-t)*1000
            row['time_to_valid_path_ms']=row['cold_total_ms']
            row['time_to_valid_path_warm_ms']=row['warm_total_ms']
        else:
            row['failure_reason']='no_validated_path' if not path else 'endpoint_tolerance_failed'
            row['metric_computation_ms']=0.0
    except Exception as exc:
        row['failure_reason']='exception'
        row['error']=f'{type(exc).__name__}: {exc}'
    return row

def write_csv(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)

def mean(values):
    a=[float(v) for v in values if v is not None and v!='']
    return statistics.mean(a) if a else None

def median(values):
    a=[float(v) for v in values if v is not None and v!='']
    return statistics.median(a) if a else None

def summarize(rows):
    grouped={}
    for row in rows: grouped.setdefault((row['algorithm'],row['scenario'],row['scenario_name']),[]).append(row)
    out=[]
    for (alg,sid,name),group in sorted(grouped.items()):
        good=[r for r in group if r['success']]
        record=dict(algorithm=alg,scenario=sid,scenario_name=name,trials=len(group),successful_trials=len(good),
                    success_rate_pct=100*len(good)/len(group),effort_unit=group[0]['effort_unit'])
        for key in ('grid_build_ms','search_ms','validation_ms','warm_total_ms','cold_total_ms','effort',
                    'validation_checks','validated_candidates','rejected_candidates'):
            record['mean_'+key+'_all']=mean(r[key] for r in group)
        for key in ('time_to_valid_path_ms','time_to_valid_path_warm_ms','path_length_m','min_clearance_m',
                    'total_turn_rad','max_turn_rad','waypoints','reference_path_length_m'):
            record['mean_'+key+'_success']=mean(r[key] for r in good)
        record['median_cold_total_ms_all']=median(r['cold_total_ms'] for r in group)
        record['median_time_to_valid_path_ms_success']=median(r['time_to_valid_path_ms'] for r in good)
        out.append(record)
    return out

def plot_summary(summary,directory):
    plots=[('success_rate_pct','Success rate (%)','success_rate.png'),
           ('mean_grid_build_ms_all','Grid build (ms)','grid_build.png'),
           ('mean_search_ms_all','Search time (ms)','search_time.png'),
           ('mean_validation_ms_all','Validation time (ms)','validation_time.png'),
           ('mean_cold_total_ms_all','Cold total (ms, all attempts)','cold_total.png'),
           ('mean_warm_total_ms_all','Warm total (ms, all attempts)','warm_total.png'),
           ('mean_time_to_valid_path_ms_success','Time to valid path (ms, successful only)','time_to_valid_path.png'),
           ('mean_path_length_m_success','Path length (m, successful only)','path_length.png'),
           ('mean_min_clearance_m_success','Minimum clearance (m, successful only)','clearance.png'),
           ('mean_total_turn_rad_success','Total heading change (rad, successful only)','turning.png'),
           ('mean_rejected_candidates_all','Rejected candidate paths','rejected_candidates.png')]
    scenarios=sorted({r['scenario'] for r in summary})
    algs=list(MODULE_PATHS)
    for key,label,filename in plots:
        fig,ax=plt.subplots(figsize=(max(9,len(scenarios)*1.35),5))
        width=.25
        for idx,alg in enumerate(algs):
            vals=[]
            for sid in scenarios:
                record=next((r for r in summary if r['algorithm']==alg and r['scenario']==sid),None)
                value=record.get(key) if record else None
                vals.append(float(value) if value is not None else float('nan'))
            ax.bar([i+(idx-1)*width for i in range(len(scenarios))],vals,width,label=alg)
        ax.set_xticks(range(len(scenarios)),scenarios,rotation=30,ha='right')
        ax.set_ylabel(label);ax.set_title(label+' by scenario');ax.legend();ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(directory/filename,dpi=150);plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description='Validated-path planner benchmark')
    parser.add_argument('--trials',type=int,default=10,help='RRT trials; use --repeat-deterministic for all planners')
    parser.add_argument('--repeat-deterministic',action='store_true')
    parser.add_argument('--scenarios',nargs='+',default=available_scenarios(),help='Scenario IDs/stems')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--rrt-iterations',type=int,default=12000)
    parser.add_argument('--dwa-steps',type=int,default=550)
    parser.add_argument('--resolution',type=float,default=0.25)
    parser.add_argument('--field-resolution',type=float,default=0.10)
    parser.add_argument('--output',type=Path,default=ROOT/'Benchmark'/'results')
    parser.add_argument("--scenario-dir", type=Path, default=ROOT / "scenarios", help="Directory containing scenario or experiment JSON files")
    args=parser.parse_args()
    if min(args.trials,args.rrt_iterations,args.dwa_steps)<=0 or min(args.resolution,args.field_resolution)<=0:
        parser.error('Counts and resolutions must be positive')
    modules={name:load_module(name,path) for name,path in MODULE_PATHS.items()}
    args.output.mkdir(parents=True,exist_ok=True)
    graphs=args.output/'graphs';graphs.mkdir(exist_ok=True)
    rows=[]
    for sid in args.scenarios:
        for alg in MODULE_PATHS:
            count=args.trials if alg=='RRT' or args.repeat_deterministic else 1
            for trial in range(count):
                seed=args.seed+sum(ord(c) for c in sid)*1000+trial
                row=run_trial(alg,modules[alg],sid,trial+1,seed,args.rrt_iterations,args.dwa_steps,args.resolution,args.field_resolution)
                rows.append(row)
                print(f'{alg:5} | {sid:15} | {trial+1:2} | {"PASS" if row["success"] else "FAIL"} | '
                      f'grid={row["grid_build_ms"]} search={row["search_ms"]} validation={row["validation_ms"]} '
                      f'cold={row["cold_total_ms"]} ms | {row["failure_reason"]} {row["error"]}',flush=True)
    summary=summarize(rows)
    write_csv(args.output/'raw_results.csv',rows,FIELDS)
    write_csv(args.output/'summary_results.csv',summary,list(summary[0]) if summary else [])
    plot_summary(summary,graphs)
    print('Saved:',args.output.resolve())

if __name__=='__main__': main()
