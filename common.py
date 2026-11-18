"""Shared map geometry, scenario registry, map layers and planner utilities.

Coordinate convention: x forward, y left, meters; fixed local-world map.
The ego vehicle collision footprint is an axis-aligned rectangle plus safety margin,
matching the original simulation approximation (NOT heading-aware).
"""
from dataclasses import dataclass, field
from pathlib import Path
import heapq
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle, Patch
from matplotlib.lines import Line2D

ROAD_LENGTH, ROAD_WIDTH, LANE_WIDTH = 50.0, 10.8, 3.6
RESOLUTION = 0.25  # Planner grid default; mapping layers can use 0.10 m.
MAP_RESOLUTION = 0.10
CAR_LENGTH, CAR_WIDTH, SAFETY_MARGIN = 4.5, 1.8, 0.25
CAR_RADIUS = math.hypot(CAR_LENGTH / 2, CAR_WIDTH / 2)
SCENARIO_DIR = Path(__file__).resolve().parent / 'scenarios'


def edges(poly):
    return list(zip(poly, poly[1:] + poly[:1]))


def point_segment_distance(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    t = max(0.0, min(1.0, ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy))) if dx*dx+dy*dy else 0
    return math.dist(p, (a[0]+t*dx, a[1]+t*dy))


def inside_polygon(p, poly):
    x, y = p
    inside = False
    for a, b in edges(poly):
        if point_segment_distance(p, a, b) < 1e-9:
            return True
        if (a[1] > y) != (b[1] > y):
            crossing = a[0] + (y-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if x < crossing:
                inside = not inside
    return inside


def segments_intersect(a, b, c, d):
    def cross(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    ab_c, ab_d, cd_a, cd_b = cross(a,b,c), cross(a,b,d), cross(c,d,a), cross(c,d,b)
    if ab_c*ab_d < 0 and cd_a*cd_b < 0:
        return True
    return any(point_segment_distance(p, u, v) < 1e-9 for p,u,v in ((c,a,b),(d,a,b),(a,c,d),(b,c,d)))


def polygons_intersect(a, b):
    return (any(inside_polygon(p,b) for p in a) or any(inside_polygon(p,a) for p in b)
            or any(segments_intersect(u,v,s,t) for u,v in edges(a) for s,t in edges(b)))


def polygon_distance(a, b):
    if polygons_intersect(a,b):
        return -1e-6
    return min([point_segment_distance(p,u,v) for p in a for u,v in edges(b)] +
               [point_segment_distance(p,u,v) for p in b for u,v in edges(a)])


def rect(cx, cy, length, width, heading=0.0):
    c, s = math.cos(heading), math.sin(heading)
    return [(cx + dx*c-dy*s, cy+dx*s+dy*c) for dx,dy in
            ((-length/2,-width/2),(length/2,-width/2),(length/2,width/2),(-length/2,width/2))]


# Fast collision primitives: the ego footprint remains axis-aligned, as in the
# original simulator. Polygon geometry is retained for mapping and visualization.
_EPS = 1e-10


def _bbox(poly):
    xs, ys = zip(*poly)
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_separation(a, b):
    return math.hypot(max(a[0]-b[2], b[0]-a[2], 0.0),
                      max(a[1]-b[3], b[1]-a[3], 0.0))


def _point_rect_distance(px, py, left, bottom, right, top):
    return math.hypot(max(left-px, px-right, 0.0), max(bottom-py, py-top, 0.0))


def _point_in_rect(p, box):
    return box[0]-_EPS <= p[0] <= box[2]+_EPS and box[1]-_EPS <= p[1] <= box[3]+_EPS


def _rect_vertices(box):
    l,b,r,t=box
    return ((l,b),(r,b),(r,t),(l,t))


def _rect_polygon_intersects(box, poly, poly_edges=None):
    # Bounding boxes reject most obstacles without any polygon work.
    if _bbox_separation(box, _bbox(poly)) > 0.0:
        return False
    corners = _rect_vertices(box)
    if any(inside_polygon(p, poly) for p in corners):
        return True
    if any(_point_in_rect(p, box) for p in poly):
        return True
    rect_edges = tuple(zip(corners, corners[1:]+corners[:1]))
    return any(segments_intersect(a,b,c,d) for a,b in rect_edges
               for c,d in (poly_edges if poly_edges is not None else edges(poly)))


def _rect_polygon_distance(box, poly, poly_edges=None):
    if _rect_polygon_intersects(box, poly, poly_edges):
        return 0.0
    corners = _rect_vertices(box)
    pe = poly_edges if poly_edges is not None else edges(poly)
    return min(min(point_segment_distance(p,a,b) for p in corners for a,b in pe),
               min(_point_rect_distance(p[0],p[1],*box) for p in poly))


def _rect_circle_distance(box, cx, cy, radius):
    return _point_rect_distance(cx,cy,*box)-radius


def _rect_rect_distance(a,b):
    dx=max(a[0]-b[2],b[0]-a[2],0.0)
    dy=max(a[1]-b[3],b[1]-a[3],0.0)
    return math.hypot(dx,dy)


@dataclass
class Road:
    boundary: list = field(default_factory=lambda: [[0,0],[50,0],[50,10.8],[0,10.8]])
    potholes: list = field(default_factory=list)
    cars: list = field(default_factory=list)
    lane_boundaries: list = field(default_factory=lambda: [3.6,7.2])
    resolution: float = MAP_RESOLUTION
    origin: tuple = (0.0,0.0)
    vehicle_length: float = CAR_LENGTH
    vehicle_width: float = CAR_WIDTH
    safety_margin: float = SAFETY_MARGIN
    metadata: dict = field(default_factory=dict)

    @property
    def bounds(self):
        xs, ys = zip(*self.boundary)
        return min(xs), min(ys), max(xs), max(ys)

    def add_pothole(self, x, y, radius=None, depth=0.1, footprint=None, confidence=1.0, severity=None, id=None):
        if footprint is None and radius is None:
            raise ValueError('Pothole requires radius or footprint')
        self.__dict__.pop('_fast_geometry', None)
        self.potholes.append({'id': id if id is not None else len(self.potholes)+1,
                              'type':'pothole', 'center':[float(x),float(y)],
                              'radius': float(radius) if radius is not None else None,
                              'footprint': [list(map(float,p)) for p in footprint] if footprint else None,
                              'depth':float(depth), 'confidence':float(confidence),
                              'severity':float(severity) if severity is not None else None})

    def add_car(self, x, y, length=CAR_LENGTH, width=CAR_WIDTH, heading=0.0, id=None):
        self.__dict__.pop('_fast_geometry', None)
        self.cars.append({'id':id if id is not None else len(self.cars)+1,
                          'center':[float(x),float(y)], 'length':float(length),
                          'width':float(width), 'heading':float(heading)})

    def ego_polygon(self, x, y):
        return rect(x,y,self.vehicle_length+2*self.safety_margin,
                    self.vehicle_width+2*self.safety_margin)

    def _geometry(self):
        """Cache fixed geometry and bounding boxes; rebuild after add_* calls."""
        cache = self.__dict__.get('_fast_geometry')
        if cache is not None:
            return cache
        boundary = self.boundary
        bb = _bbox(boundary)
        # Four-corner axis-aligned rectangles admit the original O(1) road check.
        is_rectangle = (len(boundary) == 4 and
                        set(map(tuple, boundary)) == {(bb[0],bb[1]),(bb[2],bb[1]),
                                                       (bb[2],bb[3]),(bb[0],bb[3])})
        holes=[]
        for hole in self.potholes:
            if hole['footprint']:
                poly=hole['footprint']
                holes.append(('polygon',_bbox(poly),poly,tuple(edges(poly))))
            else:
                cx,cy=hole['center']; radius=hole['radius']
                holes.append(('circle',(cx-radius,cy-radius,cx+radius,cy+radius),
                              (cx,cy,radius),None))
        cars=[]
        for car in self.cars:
            cx,cy=car['center']; length=car['length']; width=car['width']
            if abs(math.sin(car['heading'])) < 1e-12:
                box=(cx-length/2,cy-width/2,cx+length/2,cy+width/2)
                cars.append(('rectangle',box,box,None))
            else:
                poly=rect(cx,cy,length,width,car['heading'])
                cars.append(('polygon',_bbox(poly),poly,tuple(edges(poly))))
        cache=(bb,is_rectangle,tuple(edges(boundary)),holes,cars)
        self._fast_geometry=cache
        return cache

    def _ego_box(self,x,y):
        hx=self.vehicle_length/2+self.safety_margin
        hy=self.vehicle_width/2+self.safety_margin
        return x-hx,y-hy,x+hx,y+hy

    def _road_clearance(self,box,geometry):
        bounds,is_rectangle,boundary_edges,_,_=geometry
        if is_rectangle:
            return min(box[0]-bounds[0],box[1]-bounds[1],
                       bounds[2]-box[2],bounds[3]-box[3])
        corners=_rect_vertices(box)
        if not all(inside_polygon(p,self.boundary) for p in corners):
            return -1e-6
        rect_edges=tuple(zip(corners,corners[1:]+corners[:1]))
        if any(segments_intersect(a,b,c,d) for a,b in rect_edges for c,d in boundary_edges):
            return -1e-6
        return min(point_segment_distance(p,a,b) for p in corners for a,b in boundary_edges)

    def clearance(self,x,y):
        """Signed vehicle clearance using arithmetic; no Shapely construction.

        Negative means collision. Vehicle is axis-aligned (original planner model).
        """
        geometry=self._geometry()
        box=self._ego_box(x,y)
        distance=self._road_clearance(box,geometry)
        if distance < 0:
            return distance
        for kind,bounds,obj,poly_edges in geometry[3]+geometry[4]:
            # AABB separation is a LOWER bound on true separation, so if it is
            # already >= current minimum, no detailed geometry is needed.
            if _bbox_separation(box,bounds) >= distance:
                continue
            if kind=='circle':
                d=_rect_circle_distance(box,*obj)
            elif kind=='rectangle':
                d=_rect_rect_distance(box,obj)
                if d==0.0:
                    return -1e-6
            else:
                d=_rect_polygon_distance(box,obj,poly_edges)
                if d==0.0:
                    return -1e-6
            if d < 0:
                return d
            distance=min(distance,d)
        return distance

    def free(self,x,y):
        """Fast boolean collision test; avoids computing unnecessary distances."""
        geometry=self._geometry()
        box=self._ego_box(x,y)
        bounds,is_rectangle,boundary_edges,holes,cars=geometry
        if is_rectangle:
            if box[0]<bounds[0] or box[1]<bounds[1] or box[2]>bounds[2] or box[3]>bounds[3]:
                return False
        else:
            if self._road_clearance(box,geometry)<0:
                return False
        for kind,obounds,obj,poly_edges in holes+cars:
            if _bbox_separation(box,obounds)>0:
                continue
            if kind=='circle':
                if _rect_circle_distance(box,*obj)<0:
                    return False
            elif kind=='rectangle':
                if _rect_rect_distance(box,obj)==0:
                    return False
            elif _rect_polygon_intersects(box,obj,poly_edges):
                return False
        return True

    def traffic_cost(self,x,y,radius=4.0):
        cost=0.0
        for kind,bounds,obj,poly_edges in self._geometry()[4]:
            if kind=='rectangle':
                d=_point_rect_distance(x,y,*obj)
            else:
                d=point_to_polygon_distance((x,y),obj)
            cost=max(cost,max(0.0,1.0-d/radius))
        return cost

    def severity_at(self,x,y):
        severity=0.0
        for hole in self.potholes:
            cx,cy=hole['center']
            hit=(inside_polygon((x,y),hole['footprint']) if hole['footprint'] else
                 math.hypot(x-cx,y-cy)<=hole['radius'])
            if hit:
                value=hole['severity'] if hole['severity'] is not None else min(1.0,hole['depth']/0.20)
                severity=max(severity,value)
        return severity

    def grid_layers(self,resolution=None):
        """Generate 2D layers; occupancy uses POINT occupancy, not inflated ego footprint.
        Planning collision checks must continue to use road.free().
        """
        r = float(resolution or self.resolution)
        if r <= 0:
            raise ValueError('resolution must be positive')
        xmin,ymin,xmax,ymax=self.bounds
        cols,rows=math.ceil((xmax-xmin)/r),math.ceil((ymax-ymin)/r)
        drivable=np.zeros((rows,cols),dtype=bool)
        occupancy=np.zeros((rows,cols),dtype=np.uint8)  # 0 unknown, 1 free, 2 occupied
        depression=np.zeros((rows,cols),dtype=np.float32)
        severity=np.zeros((rows,cols),dtype=np.float32)
        for j in range(rows):
            y=ymin+(j+.5)*r
            for i in range(cols):
                x=xmin+(i+.5)*r
                if not inside_polygon((x,y),self.boundary):
                    occupancy[j,i]=2
                    continue
                drivable[j,i]=True
                occupancy[j,i]=1
                for hole in self.potholes:
                    cx,cy=hole['center']
                    if (inside_polygon((x,y),hole['footprint']) if hole['footprint'] else math.hypot(x-cx,y-cy)<=hole['radius']):
                        depression[j,i]=max(depression[j,i],hole['depth'])
                        severity[j,i]=max(severity[j,i],hole['severity'] if hole['severity'] is not None else min(1.0,hole['depth']/.20))
                for car in self.cars:
                    cx,cy=car['center']
                    if inside_polygon((x,y),rect(cx,cy,car['length'],car['width'],car['heading'])):
                        occupancy[j,i]=2
        traversal_cost=1.0+10.0*severity
        traversal_cost[occupancy==2]=np.inf
        return {'origin':(xmin,ymin),'resolution':r,'drivable':drivable,
                'occupancy':occupancy,'depression':depression,'severity':severity,
                'traversal_cost':traversal_cost}


def point_to_polygon_distance(p,poly):
    if inside_polygon(p,poly):
        return 0.0
    return min(point_segment_distance(p,a,b) for a,b in edges(poly))


def scenario(number, scenario_dir=None):
    directory = (
        Path(scenario_dir)
        if scenario_dir is not None
        else SCENARIO_DIR
    )
    name = str(number)
    path = directory / (
        name if name.endswith(".json") else name + ".json"
    )
    
    if not path.is_file():
        raise ValueError(f'Unknown scenario {number!r}. Available: {available_scenarios()}')
    data=json.loads(path.read_text())
    m=data.get('map',{})
    road_data=data.get('road',{})
    road=Road(boundary=road_data.get('boundary',[[0,0],[50,0],[50,10.8],[0,10.8]]),
              lane_boundaries=road_data.get('lane_boundaries',[3.6,7.2]),
              resolution=m.get('resolution',MAP_RESOLUTION),
              origin=tuple(m.get('origin',[0,0])),
              vehicle_length=data.get('vehicle',{}).get('length',CAR_LENGTH),
              vehicle_width=data.get('vehicle',{}).get('width',CAR_WIDTH),
              safety_margin=data.get('vehicle',{}).get('safety_margin',SAFETY_MARGIN),
              metadata={'coordinate_frame':m.get('coordinate_frame','local_world'),
                        'source':data.get('source','simulation')})
    for h in data.get('potholes',[]):
        cx,cy=h['center']
        road.add_pothole(cx,cy,h.get('radius'),h.get('depth',0.1),h.get('footprint'),
                         h.get('confidence',1.0),h.get('severity'),h.get('id'))
    for c in data.get('traffic',[]):
        road.add_car(*c['center'],c.get('length',CAR_LENGTH),c.get('width',CAR_WIDTH),
                     c.get('heading',0.0),c.get('id'))
    start=tuple(data['vehicle']['position'])
    goal=tuple(data['goal'])
    return road,start,goal,data.get('name',name)


def available_scenarios():
    return sorted(p.stem for p in SCENARIO_DIR.glob('*.json'))


def segment_free(road,a,b,step=0.12):
    distance=math.dist(a,b)
    return all(road.free(a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])) for t in
               np.linspace(0,1,max(2,math.ceil(distance/step)+1)))


def reference_astar(road,start,goal,resolution=.5,traffic_weight=5.0):
    xmin,ymin,xmax,ymax=road.bounds
    cols,rows=math.ceil((xmax-xmin)/resolution),math.ceil((ymax-ymin)/resolution)
    def cell(p):
        return (int((p[0]-xmin)/resolution),int((p[1]-ymin)/resolution))
    def world(c):
        return (xmin+(c[0]+.5)*resolution,ymin+(c[1]+.5)*resolution)
    s,g=cell(start),cell(goal)
    if not (0<=s[0]<cols and 0<=s[1]<rows and 0<=g[0]<cols and 0<=g[1]<rows):
        return None
    if not road.free(*start) or not road.free(*goal) or not segment_free(road,start,world(s)) or not segment_free(road,world(g),goal):
        return None
    def h(c):
        return math.dist(world(c),world(g))
    queue=[(h(s),0,s)]
    best,parent={s:0.0},{}
    counter=0
    while queue:
        f,_,cur=heapq.heappop(queue)
        if f>best[cur]+h(cur)+1e-8:
            continue
        if cur==g:
            result=[goal,world(cur)]
            while cur in parent:
                cur=parent[cur]
                result.append(world(cur))
            return [start]+result[::-1]
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx==dy==0:
                    continue
                nb=(cur[0]+dx,cur[1]+dy)
                if not (0<=nb[0]<cols and 0<=nb[1]<rows):
                    continue
                if not segment_free(road,world(cur),world(nb)):
                    continue
                d=math.hypot(dx,dy)*resolution
                cost=best[cur]+d*(1+traffic_weight*road.traffic_cost(*world(nb)))
                if cost<best.get(nb,math.inf):
                    best[nb],parent[nb]=cost,cur
                    counter+=1
                    heapq.heappush(queue,(cost+h(nb),counter,nb))
    return None


def path_length(path):
    return sum(math.dist(a,b) for a,b in zip(path,path[1:])) if path else 0.0

def draw(road, start, goal, path, title, reference=None):
    fig, ax = plt.subplots(figsize=(16, 6))
    xmin, ymin, xmax, ymax = road.bounds
    road_patch = Polygon(
        road.boundary,
        closed=True,
        facecolor="#A0A0A0",
        edgecolor="black",
        linewidth=2.8,
        zorder=1
    )
    ax.add_patch(road_patch)
    if len(road.boundary) == 4:
        for y in road.lane_boundaries:
            ax.plot(
                [xmin, xmax],
                [y, y],
                linestyle=(0, (7, 4)),
                color="white",
                linewidth=3,
                alpha=0.95,
                zorder=2
            )
        lane_edges = ([ymin]+ list(road.lane_boundaries)+ [ymax])
        for i in range(len(lane_edges) - 1):
            lane_center = (
                lane_edges[i] + lane_edges[i + 1]
            ) / 2
            ax.text(
                xmax - 1.0,
                lane_center,
                f"Lane {i + 1}",
                fontsize=13,
                color="#444444",
                ha="right",
                va="center",
                zorder=5
            )

    for hole in road.potholes:
        center = hole.get("center")
        radius = hole.get("radius")
        depth = hole.get("depth")
        footprint = hole.get("footprint")

        if footprint is not None and len(footprint) >= 3:
            polygon_points = np.asarray(footprint, dtype=float)
            pothole_patch = Polygon(
                polygon_points,
                closed=True,
                facecolor="#C65300",
                edgecolor="#8B3500",
                linewidth=2,
                alpha=1.0,
                zorder=6
            )
            ax.add_patch(pothole_patch)
            # Calculate polygon center for annotation
            polygon_center = np.mean(polygon_points, axis=0)
            annotation_center = polygon_center
            shape_type = "Irregular pothole"

        elif radius is not None and center is not None:
            pothole_patch = Circle(
                center,
                radius,
                facecolor="#C65300",
                edgecolor="#8B3500",
                linewidth=2,
                alpha=1.0,
                zorder=6
            )

            ax.add_patch(pothole_patch)

            annotation_center = center

            shape_type = "Pothole"

        else:
            print("Warning: Pothole has no valid geometry:", hole)
            continue

        # ----------------------------------------------
        # Create annotation
        # ----------------------------------------------
        annotation_lines = [shape_type]

        if radius is not None:
            annotation_lines.append(
                f"radius {radius:.1f} m"
            )

        if depth is not None:
            annotation_lines.append(
                f"depth {depth:.2f} m"
            )

        annotation = "\n".join(annotation_lines)

        # ----------------------------------------------
        # Draw annotation
        # ----------------------------------------------
        ax.annotate(
            annotation,
            xy=annotation_center,
            xytext=(
                annotation_center[0] + 3.0,
                annotation_center[1] - 1.5
            ),
            fontsize=11,
            color="black",
            ha="left",
            va="center",
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                linewidth=1.5
            ),
            zorder=8
        )

    # --------------------------------------------------
    # 6. Draw ego vehicle
    # --------------------------------------------------
    ego_length = 5.0
    ego_width = 1.8

    ego_polygon = rect(
        start[0],
        start[1],
        ego_length,
        ego_width,
        0.0
    )

    ax.add_patch(
        Polygon(
            ego_polygon,
            closed=True,
            facecolor="#0055FF",
            edgecolor="#0033AA",
            linewidth=1.5,
            zorder=6
        )
    )

    # --------------------------------------------------
    # 7. Draw A* reference path
    # --------------------------------------------------
    if reference is not None and len(reference) > 0:

        reference = np.asarray(reference)

        ax.plot(
            reference[:, 0],
            reference[:, 1],
            linestyle="--",
            color="#FFD700",
            linewidth=2,
            alpha=0.8,
            label="A* reference",
            zorder=7
        )

    # --------------------------------------------------
    # 8. Draw planned path
    # --------------------------------------------------
    if path is not None and len(path) > 0:

        path = np.asarray(path)

        ax.plot(
            path[:, 0],
            path[:, 1],
            color="cyan",
            linewidth=2.5,
            label="Planned path",
            zorder=8
        )

    # --------------------------------------------------
    # 9. Draw goal
    # --------------------------------------------------
    ax.scatter(
        goal[0],
        goal[1],
        color="#00B050",
        marker="*",
        s=350,
        edgecolors="#008040",
        linewidths=1,
        zorder=9
    )

    # --------------------------------------------------
    # 10. Configure axes
    # --------------------------------------------------
    ax.set_xlim(xmin - 1.3, xmax + 1.3)
    ax.set_ylim(ymin - 0.9, ymax + 1.3)

    ax.set_xlabel(
        "X — forward distance (m)",
        fontsize=14
    )

    ax.set_ylabel(
        "Y — lateral position (m)",
        fontsize=14
    )

    ax.set_title(
        title,
        fontsize=18,
        pad=12
    )

    ax.tick_params(
        axis="both",
        labelsize=12
    )

    # Match reference image proportions
    ax.set_aspect("equal", adjustable="box")

    # --------------------------------------------------
    # 11. Create detailed legend
    # --------------------------------------------------
    legend_elements = [

        Patch(
            facecolor="#C65300",
            edgecolor="#A84400",
            label="Pothole"
        ),

        Patch(
            facecolor="#FF3030",
            edgecolor="#B00000",
            label="Traffic vehicle"
        ),

        Patch(
            facecolor="#0055FF",
            edgecolor="#0033AA",
            label="Ego vehicle"
        ),

        Line2D(
            [0], [0],
            marker="*",
            color="w",
            markerfacecolor="#00B050",
            markeredgecolor="#008040",
            markersize=16,
            label="Goal"
        )
    ]

    if path is not None and len(path) > 0:

        legend_elements.append(
            Line2D(
                [0], [0],
                color="cyan",
                linewidth=2.5,
                label="Planned path"
            )
        )

    if reference is not None and len(reference) > 0:

        legend_elements.append(
            Line2D(
                [0], [0],
                color="#FFD700",
                linestyle="--",
                linewidth=2,
                label="A* reference"
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        fontsize=12,
        frameon=False,
        columnspacing=2.5,
        handlelength=2.5
    )

    # --------------------------------------------------
    # 12. Final layout
    # --------------------------------------------------
    fig.subplots_adjust(
        bottom=0.27,
        top=0.88
    )

    return fig
