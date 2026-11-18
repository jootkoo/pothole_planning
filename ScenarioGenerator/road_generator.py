"""Generate a centerline and non-self-intersecting offset road polygon."""
import math

def generate_road(kind, rng, length=60.0, width=10.8, samples=61):
    amplitude = {'straight':0,'gentle_curve':rng.uniform(1.0,2.0),
                 'sharp_curve':rng.uniform(2.0,3.5),'s_curve':rng.uniform(1.5,3.0),
                 'variable_width':rng.uniform(0.0,1.5)}[kind]
    direction=rng.choice([-1,1]); phase=rng.uniform(-0.25,0.25)
    points=[]; widths=[]
    for i in range(samples):
        t=i/(samples-1); x=length*t
        if kind in ('gentle_curve','sharp_curve'):
            y=direction*amplitude*(t*t*(3-2*t))
        elif kind=='s_curve':
            y=direction*amplitude*math.sin(2*math.pi*t)
        elif kind=='variable_width':
            y=direction*amplitude*math.sin(math.pi*t)
        else: y=0.0
        w=width*(1-(0.20+0.07*phase)*math.sin(math.pi*t)**2) if kind=='variable_width' else width
        points.append((x,y));widths.append(w)
    left=[];right=[];tangents=[]
    for i,(x,y) in enumerate(points):
        a=points[max(0,i-1)];b=points[min(samples-1,i+1)]
        theta=math.atan2(b[1]-a[1],b[0]-a[0]);tangents.append(theta)
        nx,ny=-math.sin(theta),math.cos(theta)
        left.append([x+nx*widths[i]/2,y+ny*widths[i]/2])
        right.append([x-nx*widths[i]/2,y-ny*widths[i]/2])
    return {'type':kind,'boundary':right+left[::-1], 'centerline':points,
            'widths':widths,'tangents':tangents,'length':length,'nominal_width':width}

def position(road,t,lateral=0):
    t=max(0.0,min(1.0,t));u=t*(len(road['centerline'])-1)
    i=min(int(u),len(road['centerline'])-2);f=u-i
    a,b=road['centerline'][i:i+2]
    x=a[0]+f*(b[0]-a[0]);y=a[1]+f*(b[1]-a[1])
    theta=math.atan2(b[1]-a[1],b[0]-a[0]);width=road['widths'][i]*(1-f)+road['widths'][i+1]*f
    return (x-lateral*math.sin(theta),y+lateral*math.cos(theta),theta,width)
