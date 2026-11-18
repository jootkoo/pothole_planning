"""Planner-independent scenario metadata/features."""
import math
from .scenario_validator import signed_area

def extract(data):
    holes=data['potholes'];road=data['road']['boundary'];cars=data['traffic']
    area=abs(signed_area(road));hole_area=sum(abs(signed_area(h['footprint'])) for h in holes)
    compact=[]
    for h in holes:
        poly=h['footprint'];a=abs(signed_area(poly))
        perimeter=sum(math.dist(p,q) for p,q in zip(poly,poly[1:]+poly[:1]))
        compact.append(4*math.pi*a/(perimeter*perimeter) if perimeter else 0)
    return {'pothole_count':len(holes),'traffic_count':len(cars),'road_area_m2':round(area,4),
            'pothole_area_m2':round(hole_area,4),'pothole_area_ratio':round(hole_area/area,6) if area else 0,
            'mean_pothole_compactness':round(sum(compact)/len(compact),5) if compact else None,
            'vehicle_length_m':data['vehicle']['length'],'vehicle_width_m':data['vehicle']['width'],
            'safety_margin_m':data['vehicle']['safety_margin'],
            'start_goal_distance_m':round(math.dist(data['vehicle']['position'],data['goal']),4)}
