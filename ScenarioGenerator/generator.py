"""CLI: python -m ScenarioGenerator.generator --count 25 --seed 42"""
import argparse,csv,json,random,sys
from pathlib import Path
from .road_generator import generate_road,position
from .pothole_generator import make_shape,transform
from .placement_generator import proposals
from .traffic_generator import generate_traffic
from .vehicle_generator import DEFAULT_VEHICLES
from .scenario_validator import validate,inside
from .feature_extractor import extract

ROAD_TYPES=['straight','gentle_curve','sharp_curve','s_curve','variable_width']
SHAPES=['circular','elliptical','irregular_convex','concave','elongated','clustered']
PLACEMENTS=['random','lane_centered','alternating','clustered','narrow_passage','blockage']
TRAFFIC=['none','low','medium','high']



def _point_segment_distance(point, start, end):
    """Euclidean distance from a point to a finite line segment (meters)."""
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_squared))
    return ((px - (ax + t * dx)) ** 2 + (py - (ay + t * dy)) ** 2) ** 0.5


def _simplify_polyline(points, tolerance):
    """Douglas-Peucker simplification; always retain the two endpoints."""
    if len(points) <= 2:
        return list(points)
    start, end = points[0], points[-1]
    distances = [_point_segment_distance(pt, start, end) for pt in points[1:-1]]
    max_distance = max(distances)
    if max_distance <= tolerance:
        return [start, end]
    split = distances.index(max_distance) + 1
    return (_simplify_polyline(points[:split + 1], tolerance)[:-1]
            + _simplify_polyline(points[split:], tolerance))


def simplify_road_boundary(boundary, tolerance=0.02):
    """Simplify both road edges independently, preserving the four end corners.

    Input from road_generator.generate_road(): right edge from start to end,
    followed by left edge from end to start. No repeated closing vertex.
    This leaves straight roads with exactly four vertices, while retaining
    curved/variable-width edge points as needed. Units: meters.
    """
    if len(boundary) < 4 or len(boundary) % 2:
        raise ValueError('Expected an even-length road boundary with two edges')
    midpoint = len(boundary) // 2
    right = _simplify_polyline(boundary[:midpoint], tolerance)
    left = _simplify_polyline(boundary[midpoint:], tolerance)
    return right + left

def build_one(index,seed,config,boundary_tolerance=0.02):
    rng=random.Random(seed)
    road_type=ROAD_TYPES[index%len(ROAD_TYPES)]
    shape=SHAPES[(index//len(ROAD_TYPES))%len(SHAPES)]
    placement=PLACEMENTS[(index//(len(ROAD_TYPES)*len(SHAPES)))%len(PLACEMENTS)]
    traffic_level=TRAFFIC[(index//(len(ROAD_TYPES)*len(SHAPES)*len(PLACEMENTS)))%len(TRAFFIC)]
    road=generate_road(road_type,rng,length=rng.uniform(*config['road_length_range']),
                       width=rng.uniform(*config['road_width_range']))
    sx,sy,sh,_=position(road,.07);gx,gy,_,_=position(road,.93)
    count=rng.randint(*config['pothole_count_range'])
    if placement=='blockage':count=max(count,3)
    potholes=[]
    for t,offset in proposals(placement,count,rng):
        # Bounded retries: do not silently accept malformed/outside-road polygons.
        for attempt in range(35):
            if attempt:
                t2=max(.12,min(.88,t+rng.uniform(-.04,.04)))
                off2=max(-.34,min(.34,offset+rng.uniform(-.08,.08)))
            else:t2,off2=t,offset
            x,y,heading,width=position(road,t2,off2*road['nominal_width'])
            size=rng.uniform(*config['pothole_size_range'])
            local=make_shape(shape,rng,size)
            poly=transform(local,(x,y),heading+rng.uniform(-.8,.8))
            if all(inside(p,road['boundary']) for p in poly):break
        else:continue
        potholes.append({'id':len(potholes)+1,'center':[round(x,5),round(y,5)],
                         'footprint':poly,'depth':round(rng.uniform(.03,.30),3),
                         'confidence':round(rng.uniform(.85,1.0),3),
                         'severity':round(rng.uniform(.2,1.0),3)})
    cars=generate_traffic(road,rng,traffic_level)
    sid = f'generated_{index:06d}'
    actual_pothole_count = len(potholes)
    actual_traffic_count = len(cars)
    actual_shape = shape if actual_pothole_count else 'none'
    actual_placement = placement if actual_pothole_count else 'none'

    scenario_metadata = {
        'generator_version': '1.1.0',
        'seed': seed,
        'road': {
            'type': road_type,
            'length_m': round(road['length'], 3),
            'nominal_width_m': round(road['nominal_width'], 3)
        },
        'potholes': {
            'selected_shape_type': shape,
            'shape_type': actual_shape,
            'requested_count': count,
            'actual_count': actual_pothole_count
        },
        'placement': {
            'selected_strategy': placement,
            'strategy': actual_placement
        },
        'traffic': {
            'requested_level': traffic_level,
            'actual_count': actual_traffic_count
        }
    }

    original_boundary = road['boundary']
    simplified_boundary = simplify_road_boundary(original_boundary, boundary_tolerance) if boundary_tolerance is not None else original_boundary
    scenario_metadata['road']['boundary_vertices_original'] = len(original_boundary)
    scenario_metadata['road']['boundary_vertices_saved'] = len(simplified_boundary)
    scenario_metadata['road']['boundary_simplification_tolerance_m'] = boundary_tolerance

    data = {
        'id': sid,
        'name': f'{road_type} / {actual_shape} / {actual_placement} / {traffic_level}',
        'source': 'synthetic',
        'metadata': scenario_metadata,
        'map': {'coordinate_frame': 'local_world', 'origin': [0, 0],
                'resolution': .1, 'units': 'meters'},
        'road': {'boundary': [[round(v, 5) for v in pt] for pt in simplified_boundary],
                 'lane_boundaries': []},
        'vehicle': {'position': [round(sx, 5), round(sy, 5)],
                    'length': 4.5, 'width': 1.8, 'safety_margin': .25,
                    'heading': sh, 'speed': 0},
        'goal': [round(gx, 5), round(gy, 5)],
        'potholes': potholes,
        'traffic': cars
    }
    errors = validate(data)
    if errors:
        raise ValueError(f'{sid}: {errors}')

    meta = {
        'scenario_id': sid,
        'seed': seed,
        'generator_version': '1.1.0',
        'road_type': road_type,
        'road_length_m': round(road['length'], 3),
        'nominal_road_width_m': round(road['nominal_width'], 3),
        'pothole_type': actual_shape,
        'selected_pothole_type': shape,
        'requested_potholes': count,
        'placed_potholes': actual_pothole_count,
        'placement': actual_placement,
        'selected_placement': placement,
        'traffic_level': traffic_level,
        'traffic_count': actual_traffic_count,
        'boundary_vertices_original': len(original_boundary),
        'boundary_vertices_saved': len(simplified_boundary),
        'boundary_simplification_tolerance_m': boundary_tolerance
    }
    return data,meta

def generate(count,seed,output,config,vehicles=None,boundary_tolerance=0.02):
    output=Path(output);(output/'scenarios').mkdir(parents=True,exist_ok=True)
    (output/'experiments').mkdir(exist_ok=True)
    vehicles=vehicles or DEFAULT_VEHICLES
    metadata=[];features=[];experiments=[]
    for i in range(count):
        scenario_seed=seed+i*104729
        data,meta=build_one(i,scenario_seed,config,boundary_tolerance=boundary_tolerance)
        sid=data['id'];(output/'scenarios'/f'{sid}.json').write_text(json.dumps(data,indent=2)+'\n')
        metadata.append(meta)
        for vehicle in vehicles:
            experiment=json.loads(json.dumps(data))
            experiment['id']=f'{sid}_{vehicle["id"]}'
            experiment['experiment_metadata'] = {'scenario_id': sid, 'vehicle_id': vehicle['id']}
            experiment['vehicle'].update({k:v for k,v in vehicle.items() if k!='id'})
            errors=validate(experiment)
            if errors:raise ValueError(errors)
            filename=f'{experiment["id"]}.json'
            (output/'experiments'/filename).write_text(json.dumps(experiment,indent=2)+'\n')
            features.append({'experiment_id':experiment['id'],'scenario_id':sid,
                             'vehicle_id':vehicle['id'],**meta,**extract(experiment)})
            experiments.append({'experiment_id':experiment['id'],'scenario_id':sid,
                                'vehicle_id':vehicle['id'],'scenario_file':f'experiments/{filename}'})
    for filename,rows in [('metadata.csv',metadata),('features.csv',features),('experiments.csv',experiments)]:
        if rows:
            with (output/filename).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'manifest.json').write_text(json.dumps({'seed':seed,'environments':count,
        'vehicle_variants':len(vehicles),'experiments':len(experiments),
        'boundary_simplification_tolerance_m':boundary_tolerance,'config':config},indent=2)+'\n')
    return len(experiments)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--count',type=int,default=25);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--output',default='generated');ap.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    ap.add_argument('--boundary-tolerance',type=float,default=0.02,
                    help='Boundary simplification tolerance in meters (default: 0.02)')
    ap.add_argument('--no-simplify-boundary',action='store_true',
                    help='Save original dense boundary for baseline benchmarking')
    args=ap.parse_args()
    if args.boundary_tolerance < 0:ap.error('--boundary-tolerance must be >= 0')
    if args.count<1:ap.error('--count must be >= 1')
    config=json.loads(Path(args.config).read_text())
    n=generate(args.count,args.seed,args.output,config,
               boundary_tolerance=None if args.no_simplify_boundary else args.boundary_tolerance)
    print(f'Generated {args.count} road environments and {n} vehicle experiments in {args.output}')
if __name__=='__main__':main()
