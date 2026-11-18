"""Generate stationary traffic obstacles aligned to local road tangents."""
from .road_generator import position

def generate_traffic(road,rng,level):
    count={'none':0,'low':rng.randint(0,1),'medium':rng.randint(1,2),'high':rng.randint(2,4)}[level]
    traffic=[]
    for i in range(count):
        t=rng.uniform(.20,.82);lateral=rng.choice([-.28,0,.28])
        x,y,heading,_=position(road,t,lateral*road['nominal_width'])
        traffic.append({'id':i+1,'center':[round(x,5),round(y,5)],
                        'length':round(rng.uniform(3.7,5.5),3),
                        'width':round(rng.uniform(1.6,2.1),3),'heading':heading})
    return traffic
