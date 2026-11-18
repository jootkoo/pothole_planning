"""Construct valid simple polygon pothole shapes in local coordinates."""
import math

def make_shape(kind,rng,size):
    n=rng.randint(9,16)
    if kind=='circular': n=16
    if kind=='clustered': kind=rng.choice(['elliptical','irregular_convex','concave'])
    angles=[2*math.pi*i/n for i in range(n)]
    if kind=='circular': radial=[1.0]*n;ax=ay=size
    elif kind=='elliptical': radial=[1.0]*n;ax=size*rng.uniform(1.2,1.8);ay=size*rng.uniform(.55,.9)
    elif kind=='elongated': radial=[rng.uniform(.9,1.05) for _ in angles];ax=size*rng.uniform(1.7,2.4);ay=size*rng.uniform(.35,.55)
    elif kind=='irregular_convex':
        # A convex hull is computed below; random radial points alone are not necessarily convex.
        radial=[rng.uniform(.72,1.20) for _ in angles];ax=ay=size
    elif kind=='concave': radial=[(.43 if i%4==0 else rng.uniform(.9,1.12)) for i in range(n)];ax=ay=size
    else: raise ValueError(kind)
    pts=[(ax*r*math.cos(a),ay*r*math.sin(a)) for r,a in zip(radial,angles)]
    if kind=='irregular_convex': pts=convex_hull(pts)
    return pts

def convex_hull(points):
    points=sorted(set(points))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[];upper=[]
    for q in points:
        while len(lower)>=2 and cross(lower[-2],lower[-1],q)<=0:lower.pop()
        lower.append(q)
    for q in reversed(points):
        while len(upper)>=2 and cross(upper[-2],upper[-1],q)<=0:upper.pop()
        upper.append(q)
    return lower[:-1]+upper[:-1]

def transform(shape,center,rotation):
    c,s=math.cos(rotation),math.sin(rotation)
    return [[round(center[0]+c*x-s*y,5),round(center[1]+s*x+c*y,5)] for x,y in shape]
