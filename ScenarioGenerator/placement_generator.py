"""Generate reproducible road-relative obstacle placement proposals."""
def proposals(strategy,count,rng):
    for i in range(count):
        if strategy=='alternating':
            t=.18+.65*(i+.5)/max(count,1);lateral=(1 if i%2 else -1)*rng.uniform(.18,.38)
        elif strategy=='lane_centered':
            t=rng.uniform(.12,.88);lateral=rng.choice([-.31,0,.31])+rng.uniform(-.04,.04)
        elif strategy=='clustered':
            t=max(.13,min(.87,.48+rng.uniform(-.13,.13)));lateral=rng.uniform(-.36,.36)
        elif strategy=='narrow_passage':
            t=.40+.13*(i//2)+rng.uniform(-.015,.015);lateral=(1 if i%2 else -1)*rng.uniform(.25,.37)
        elif strategy=='blockage':
            t=.50+rng.uniform(-.025,.025);lateral=(-.34+(i%3)*.34)+rng.uniform(-.025,.025)
        else:
            t=rng.uniform(.12,.88);lateral=rng.uniform(-.37,.37)
        yield max(.10,min(.90,t)),max(-.43,min(.43,lateral))
