"""Check input geometry; intentionally DO NOT require a feasible route."""
import math

def signed_area(poly):
    return sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1]))/2

def _orient(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def _crosses(a,b,c,d):
    eps=1e-8
    return _orient(a,b,c)*_orient(a,b,d)<-eps and _orient(c,d,a)*_orient(c,d,b)<-eps

def simple(poly):
    n=len(poly)
    if n<3 or abs(signed_area(poly))<1e-6:return False
    for i in range(n):
        for j in range(i+1,n):
            if abs(i-j)<=1 or (i==0 and j==n-1):continue
            if _crosses(poly[i],poly[(i+1)%n],poly[j],poly[(j+1)%n]):return False
    return True

def inside(pt,poly):
    x,y=pt;v=False
    for a,b in zip(poly,poly[1:]+poly[:1]):
        if (a[1]>y)!=(b[1]>y) and x<a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]):v=not v
    return v

def validate(data,require_obstacles_inside=True):
    errors=[];boundary=data['road']['boundary']
    if not simple(boundary):errors.append('road boundary is not a simple nonzero-area polygon')
    if not inside(data['vehicle']['position'],boundary):errors.append('start outside road')
    if not inside(data['goal'],boundary):errors.append('goal outside road')
    for h in data['potholes']:
        poly=h['footprint']
        if not simple(poly):errors.append(f"pothole {h['id']} is invalid")
        elif require_obstacles_inside and not all(inside(p,boundary) for p in poly):errors.append(f"pothole {h['id']} extends outside road")
    # Vehicle footprint and obstacle overlaps are checked by planners, not this geometry-only check.
    return errors
