# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 08:19:21 2022

@author: ar2806
"""

import random
import numpy as np
import pandas as pd


CHROMOSOME_SPECS = {
    3: [7, 17, 17],
    5: [7, 7, 17, 17, 17],
    7: [7, 7, 7, 17, 17, 17, 17],
}


def repair_chromosome(chromosome):
    """Keep every gene inside the valid range for its chromosome position.

    Binary activation genes must refer to b0-b6, while unary activation genes
    must refer to f0-f16. Crossover can move a unary gene into a binary slot,
    so every generated chromosome is repaired before being evaluated.
    """
    repaired = [int(gene) for gene in chromosome]
    if len(repaired) not in CHROMOSOME_SPECS:
        raise ValueError(f"Unsupported chromosome length: {len(repaired)}")

    limits = CHROMOSOME_SPECS[len(repaired)]
    return [gene % limit for gene, limit in zip(repaired, limits)]


def create_initial_pop(pop_size=30):
    #arg pop is the array of the shape of [no of binary function, no of unary function, no unary function] like [3,4,4]
    #arg pop_size is the number of members in a single generation, like 10,20,30 or 50
    #returns a random population that contains 0 and 1
    
    final_population =[]
    for i in range(pop_size):
        chromosome_len = [3,5,7]
        pop_len = random.choice(chromosome_len)
        pop = CHROMOSOME_SPECS[pop_len]
        single_pop=[]
        for j in range(len(pop)):
            single_pop.append(random.randint(0, pop[j]-1))
        final_population.append(repair_chromosome(single_pop))
    return final_population



def crossover(a,b): #one point cross-over
    new_a = a
    new_b = b
    prob = random.uniform(0,1)
    if prob<0.8:
        a_len = len(a)
        b_len = len(b)
        if a_len==b_len:
            site = random.randint(0,a_len-1)
        elif a_len>b_len:
            site = random.randint(0,b_len-1)
        else:
            site = random.randint(0,a_len-1)
        new_a = [*a[:site], *b[site:]]
        new_b = [*b[:site], *a[site:]]
    return repair_chromosome(new_a), repair_chromosome(new_b)



def mutation(a):
    pop_len = len(a)
    prob = random.uniform(0,1)
    unary_replace = random.randint(0,16)
    binary_replace = random.randint(0,6)
    if prob>0.3:
        if pop_len==3:
            loc = random.randint(0,2)
            if loc==0:
                a[0]= binary_replace
            elif loc==1:
                a[1]= unary_replace
            else:
                a[2]= unary_replace
        if pop_len==5:
            loc = random.randint(0,4)
            if loc==0:
                a[0]= binary_replace
            elif loc==1:
                a[1]= binary_replace
            elif loc==2:
                a[2]= unary_replace
            elif loc==3:
                a[3]= unary_replace
            else:
                a[4]= unary_replace
        if pop_len==7:
            loc = random.randint(0,6)
            if loc==0:
                a[0]= binary_replace
            elif loc==1:
                a[1]= binary_replace
            elif loc==2:
                a[2]= binary_replace
            elif loc==3:
                a[3]= unary_replace
            elif loc==4:
                a[4]= unary_replace
            elif loc==5:
                a[5]= unary_replace
            else:
                a[6]= unary_replace
    return repair_chromosome(a)            


def _row_to_chromosome(row):
    chromosome = []
    for value in row:
        if pd.isna(value):
            continue
        chromosome.append(int(value))
    return repair_chromosome(chromosome)


def _ensure_population_size(population, target_size):
    if len(population) < target_size:
        population.extend(create_initial_pop(target_size - len(population)))
    return population[:target_size]


def next_generation(population, scores, pop_size=30):#input arg is a dataframe containing population ans scores
    #need to take the top 2 best scoring chromosome and save it to another dataframe for passing them to crossover
    #call the crossover function to get the new 2 chromosomes
    #call the create initial population function to generate 6 random chromosome
    # so, best 2 + crossover 2 + random new 6 = 10 will be the new generation
    population = pd.DataFrame(population)
    population['score']= scores
    population.sort_values(by= ['score'], ascending = False, inplace=True)
    population.drop(['score'], inplace=True, axis=1, errors='ignore')
    population.drop_duplicates(subset=None, keep='first', inplace=True)
    if population.empty:
        return create_initial_pop(pop_size)

    elite_count = min(max(2, pop_size // 5), len(population.index))
    best_rows = population.head(elite_count).values.tolist()
    best_pop = [_row_to_chromosome(row) for row in best_rows]

    next_pop = [repair_chromosome(chromosome[:]) for chromosome in best_pop]

    crossover_pool = best_pop[:]
    if len(crossover_pool) % 2 != 0:
        crossover_pool = crossover_pool[:-1]
    for i in range(0, len(crossover_pool), 2):
        new_a, new_b = crossover(crossover_pool[i][:], crossover_pool[i + 1][:])
        next_pop.append(new_a)
        next_pop.append(new_b)

    mut_candidate = random.choice(best_pop)[:]
    next_pop.append(mutation(mut_candidate))

    random_needed = pop_size - len(next_pop)
    if random_needed > 0:
        next_pop.extend(create_initial_pop(random_needed))

    best_pop = _ensure_population_size(next_pop, pop_size)
    print('Passing to Next Generation')
    return best_pop
