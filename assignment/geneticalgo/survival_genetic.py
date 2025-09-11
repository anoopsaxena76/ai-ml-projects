import random

def maximize_survival(seed: int = 123):
    #Author:Anoop K. Saxena
    #Purpose:Single function that (a) defines the backpack problem,
    #        (b) creates 10 random chromosomes, and (c) maximizes the
    #Each item:(name, weight, survival_value, penalty_if_NOT_taken)
    ITEMS=[
        ("Sleeping Bag",30,20,  0),
        ("Rope",        10,10,  0),
        ("Bottle",       5,20,  0),
        ("Torch+Battery",15,25,-20),  #essential
        ("Glucose",      5,30,  0),
        ("Pocket Knife",10,15,-10),   #essential
        ("Umbrella",    20,10,  0),
    ]
    CAPACITY=40  #kg

    #Derived constants for relative scoring (fixed, not exposed)
    MAX_VALUE   = sum(v for _,_,v,_ in ITEMS)            #sum of all values
    MAX_PENALTY = sum(-p for *_,p in ITEMS if p<0)       #sum of abs essential penalties

    def fitness(bits):
        #Compute normalized objective:
        #f = (value/MAX_VALUE) - (weight/CAPACITY) - (abs(penalty)/MAX_PENALTY)
        w=0; val=0; pen=0
        for b,(_,wt,v,p) in zip(bits,ITEMS):
            if b: 
                w+=wt; val+=v
            else:
                pen+=p
        rel_value   = (val/MAX_VALUE)   if MAX_VALUE>0   else 0.0
        rel_weight  = (w/CAPACITY)      if CAPACITY>0    else 0.0
        rel_penalty = (abs(pen)/MAX_PENALTY) if MAX_PENALTY>0 else 0.0
        f = rel_value - rel_weight - rel_penalty
        return f, {"w":w, "val":val, "pen":pen,
                   "rel_value":rel_value, "rel_weight":rel_weight, "rel_penalty":rel_penalty}

    def crossover(a,b):
        #Single-point crossover (fixed behavior, no external knobs)
        n=len(a)
        if n<=1: return a[:],b[:]
        cut=random.randint(1,n-1)
        return a[:cut]+b[cut:], b[:cut]+a[cut:]

    def mutate(bits):
        #Bit-flip mutation with a small fixed probability (0.05)
        for i in range(len(bits)):
            if random.random()<0.05:
                bits[i]=1-bits[i]

    def tournament(pop):
        #Tournament selection with fixed size k=3
        k=3
        cand=random.sample(pop,k)
        return max(cand, key=lambda c: fitness(c)[0])[:]

    #-----------------------------
    #Core: create 10 random chromosomes and evolve briefly
    #-----------------------------
    random.seed(seed)
    n=len(ITEMS)

    #Only this part is not hardcoded: generate 10 random chromosomes
    population=[[random.randint(0,1) for _ in range(n)] for _ in range(10)]

    #Track best-so-far
    best = max(population, key=lambda c: fitness(c)[0])
    best_fit,_ = fitness(best)

    #Fixed tiny GA cycle (no parameters exposed): 120 generations, pc=0.90
    for _ in range(120):
        new=[]
        elite=best[:]  #elitism

        while len(new)<10:
            p1=tournament(population)
            p2=tournament(population)
            if random.random()<0.90:
                c1,c2=crossover(p1,p2)
            else:
                c1,c2=p1[:],p2[:]
            mutate(c1); mutate(c2)
            new.extend([c1,c2])

        population=new[:10]

        #Inject elite: replace worst
        worst_i=min(range(10), key=lambda i: fitness(population[i])[0])
        population[worst_i]=elite

        #Update best
        cand=max(population, key=lambda c: fitness(c)[0])
        cand_fit,_=fitness(cand)
        if cand_fit>best_fit:
            best,best_fit=cand[:],cand_fit

    #Prepare return values
    bits="".join(map(str,best))
    score,details=fitness(best)
    return bits, score, details
