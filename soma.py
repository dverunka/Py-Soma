import numpy
from numpy import asarray


# individual of the population. It holds parameters and fitness of the solution
class Individual:
    def __init__(self, params, fitness):
        self.params = params
        self.fitness = fitness

    def __repr__(self):
        return 'params: {} fitness: {}'.format(self.params, self.fitness)


# class holding all the result information
class Result:
    def __init__(self, 
                 benchmark_function_name: str,
                 iteration: int,
                 dimensions: int,
                 individual: Individual):
        self.benchmark_function_name = benchmark_function_name
        self.iteration = iteration
        self.dimensions = dimensions
        self.individual = individual


# benchmark cost functions:

def de_jong_first(params):
    return numpy.sum(numpy.square(params))

def de_jong_second(params):
    assert len(params) >= 2
    params = asarray(params)
    return numpy.sum(100 * (params[1:] - params[:-1] ** 2) ** 2 + (1 - params[:-1]) ** 2)

def rastrigin(params):
    return 10 * len(params) + numpy.sum(numpy.square(params) - 10 * numpy.cos(2 * numpy.pi * params))

def schwefel(params):
    return 418.9829 * len(params) - numpy.sum(params * numpy.sin(numpy.sqrt(numpy.abs(params))))


# return fitness of the params
def evaluate(params):
    #return de_jong_first(params)
    return de_jong_second(params)
    #return schwefel(params)    
    #return rastrigin(params)


# reassign params that are outside the bounds by a random number within the bounds
def bounded(params, min_s: list, max_s: list):
    return numpy.array([numpy.random.uniform(min_s[d], max_s[d])
            if params[d] < min_s[d] or params[d] > max_s[d] 
            else params[d] 
            for d in range(len(params))])

# generate min bounds array
def generate_min_s(dimensions):
    return [-10] * dimensions

# generate max bounds array
def generate_max_s(dimensions):
    return [10] * dimensions

# generate individual params
def generate_individual():
    params = numpy.random.uniform(min_s, max_s, dimensions)
    fitness = evaluate(params)
    return Individual(params, fitness)


# generate initial population
def generate_population(size, min_s, max_s, dimensions):
    return [generate_individual() for _ in range(size)]


def generate_prt_vector(prt, dimensions):
    return numpy.random.choice([0, 1], dimensions, p=[prt, 1-prt])


# find leader of the population by its fitness (the lower the better)
def get_leader(population):
    return min(population, key = lambda individual: individual.fitness)


# SOMA all-to-one algorithm
def soma_all_to_one(population, prt, path_length, step, fes, min_s, max_s, dimensions):
    fes_iteration = 0
    while fes_iteration < fes: # check fes
        leader = get_leader(population)

        for individual in population:
            if fes_iteration >= fes: # check fes
                break

            if individual is leader:
                continue

            next_position = individual.params
            prt_vector = generate_prt_vector(prt, dimensions)

            for t in numpy.arange(step, path_length, step):
                if fes_iteration >= fes: # check fes
                    break

                current_position = individual.params + (leader.params - individual.params) * t * prt_vector
                current_position = bounded(current_position, min_s, max_s)
                fitness = evaluate(current_position)
                
                fes_iteration += 1 # increment fes iteration

                if fitness <= individual.fitness:
                    next_position = current_position
                    individual.fitness = fitness

            individual.params = next_position

    return get_leader(population)


###


# SOMA parameters
pop_size = 50
prt = 0.3
path_lenght = 3
step = 0.33

# general parameters
dimensions = 10
fes = 5000 * dimensions
iterations = 30
min_s = generate_min_s(dimensions)
max_s = generate_max_s(dimensions)

# run
population = generate_population(pop_size, min_s, max_s, dimensions)
result = soma_all_to_one(population, prt, path_lenght, step, fes, min_s, max_s, dimensions)
print(result)