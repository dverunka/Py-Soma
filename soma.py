import numpy
import matplotlib.pyplot as plt


# SOMA parameters
pop_size = 50
prt = 0.3
path_lenght = 3
step = 0.33
iterations = 3 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# individual of the population. It holds parameters and fitness of the solution
class Individual:
    def __init__(self, params, fitness):
        self.params = params
        self.fitness = fitness

    def __repr__(self):
        return 'params: {} fitness: {}'.format(self.params, self.fitness)


# benchmark cost functions:

def de_jong_first(params): # -5..5
    return numpy.sum(numpy.square(params))

def de_jong_second(params): # -2..2
    return numpy.sum(100 * numpy.square(numpy.square(params[:-1]) - params[1:]) + numpy.square(1 - params[:-1]))

def rastrigin(params): # -2..2
    return 2 * len(params) + numpy.sum(numpy.square(params) - 10 * numpy.cos(2 * numpy.pi * params))

def schwefel(params): # 0..500
    return numpy.sum((-1) * params * numpy.sin(numpy.sqrt(numpy.abs(params))))


# reassign params that are outside the bounds by a random number within the bounds
def bounded(params, min_s: list, max_s: list):
    return numpy.array([numpy.random.uniform(min_s[d], max_s[d])
            if params[d] < min_s[d] or params[d] > max_s[d] 
            else params[d] 
            for d in range(len(params))])

# generate min bounds array
def generate_min_s(dimensions, min_border):
    return [min_border] * dimensions

# generate max bounds array
def generate_max_s(dimensions, max_border):
    return [max_border] * dimensions

# generate individual params
def generate_individual(benchmark, min_s, max_s, dimensions):
    params = numpy.random.uniform(min_s, max_s, dimensions)
    fitness = benchmark(params)
    return Individual(params, fitness)

# generate initial population
def generate_population(size, min_s, max_s, dimensions, benchmark):
    return [generate_individual(benchmark, min_s, max_s, dimensions) for _ in range(size)]

def generate_prt_vector(prt, dimensions):
    return numpy.random.choice([0, 1], dimensions, p=[prt, 1-prt])

# find leader of the population by its fitness (the lower the better)
def get_leader(population):
    return min(population, key = lambda individual: individual.fitness)

# set graph params
def set_graph(title, min_x, max_x, min_y, max_y):
    plt.xlabel("FES")
    plt.ylabel("fitness")
    plt.grid(True)
    plt.axis((min_x, max_x, min_y, max_y))
    plt.title(title)

# SOMA all-to-one algorithm
def soma_all_to_one(population, prt, path_length, step, fes, min_s, max_s, dimensions, benchmark):
    fes_iteration = 0
    history_fitness_list = []
    global_min_fitness = numpy.Inf

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
                fitness = benchmark(current_position)
                
                fes_iteration += 1 # increment fes iteration

                if fitness <= individual.fitness:
                    next_position = current_position
                    individual.fitness = fitness

                if fitness < global_min_fitness:
                    global_min_fitness = fitness

                history_fitness_list.append(global_min_fitness)

            individual.params = next_position

    return get_leader(population), history_fitness_list


# run iterations of SOMA algorithm
def run_algorithm(dimensions, min_border, max_border, benchmark):
    fes = 5000 * dimensions
    min_s = generate_min_s(dimensions, min_border)
    max_s = generate_max_s(dimensions, max_border)

    fitness_list = []
    list_of_history_fitness_lists = []

    for i in range(iterations):
        population = generate_population(pop_size, min_s, max_s, dimensions, benchmark)
        winner, history_fitness_list = soma_all_to_one(population, prt, path_lenght, step, fes, min_s, max_s, dimensions, benchmark)
        print(winner)
        print("")

        fitness_list.append(winner.fitness)
        list_of_history_fitness_lists.append(history_fitness_list)

    return fitness_list, list_of_history_fitness_lists


###


# run de jong first, d = 10
djf10_best_fitnesses, djf10_history_fitnesses = run_algorithm(10, -5, 5, de_jong_first)

set_graph("SOMA de jong first, d = 10", 0, 5000, 0, 50)
for hf in djf10_history_fitnesses:
    plt.plot(range(1, len(hf) + 1), hf, linewidth = 0.5)
plt.savefig("djf_10.png")
plt.clf()
plt.cla()
plt.close()

djf10_min = numpy.min(djf10_best_fitnesses)
djf10_max = numpy.max(djf10_best_fitnesses)
djf10_mean = numpy.mean(djf10_best_fitnesses)
djf10_median = numpy.median(djf10_best_fitnesses)
djf10_std = numpy.std(djf10_best_fitnesses)

dhf10_avg = numpy.mean(djf10_history_fitnesses, axis = 0)
set_graph("SOMA de jong first, d = 10, avg", 0, 5000, 0, 50)
plt.plot(range(1, len(dhf10_avg) + 1), dhf10_avg, linewidth = 1)
plt.savefig("djf_10_avg.png")
plt.clf()
plt.cla()
plt.close()

# # run de jong first, d = 30
# djf30 = run_algorithm(30, -5, 5, de_jong_first)
# djf30_min = 
# djf30_max = 
# djf30_mean = 
# djf30_median = 
# djf30_std = 

# # run de jong second, d = 10
# djs10 = run_algorithm(10, -2, 2, de_jong_second)
# djs10_min = 
# djs10_max = 
# djs10_mean = 
# djs10_median = 
# djs10_std = 
# # run de jong second, d = 30
# djs30 = run_algorithm(30, -2, 2, de_jong_second)
# djs30_min = 
# djs30_max = 
# djs30_mean = 
# djs30_median = 
# djs30_std = 

# # run rastrigin, d = 10
# ras10 = run_algorithm(10, -2, 2, rastrigin)
# res10_min = 
# res10_max = 
# res10_mean = 
# res10_median = 
# res10_std = 
# # run rastrigin, d = 30
# ras30 = run_algorithm(30, -2, 2, rastrigin)
# res30_min = 
# res30_max = 
# res30_mean = 
# res30_median = 
# res30_std = 

# # run schwefel, d = 10
# sch10 = run_algorithm(10, 0, 500, schwefel)
# sch10_min = 
# sch10_max = 
# sch10_mean = 
# sch10_median = 
# sch10_std = 
# # run schwefel, d = 30
# sch30 = run_algorithm(30, 0, 500, schwefel)
# sch30_min = 
# sch30_max = 
# sch30_mean = 
# sch30_median = 
# sch30_std = 