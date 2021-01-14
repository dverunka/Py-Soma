import numpy
import matplotlib.pyplot as plt


# SOMA parameters
pop_size = 50
prt = 0.3
path_lenght = 3
step = 0.33
iterations = 30


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
        print(f"fitness: {winner.fitness}")

        fitness_list.append(winner.fitness)
        list_of_history_fitness_lists.append(history_fitness_list)

    return fitness_list, list_of_history_fitness_lists


###

# run de jong first, d = 10
djfirst10_best_fitnesses, djfirst10_history_fitnesses = run_algorithm(10, -5, 5, de_jong_first)

set_graph("SOMA de jong first, d = 10", 0, 5000, 0, 50)
for hf in djfirst10_history_fitnesses:
    plt.plot(range(1, len(hf) + 1), hf, linewidth = 0.5)
plt.savefig("djfirst_10.png")
plt.clf()
plt.cla()
plt.close()

djfirst10_min = numpy.min(djfirst10_best_fitnesses)
djfirst10_max = numpy.max(djfirst10_best_fitnesses)
djfirst10_mean = numpy.mean(djfirst10_best_fitnesses)
djfirst10_median = numpy.median(djfirst10_best_fitnesses)
djfirst10_std = numpy.std(djfirst10_best_fitnesses)
print(f"djfirst10 min: {djfirst10_min} max: {djfirst10_max} mean: {djfirst10_mean} median: {djfirst10_median} std: {djfirst10_std}")

djfirst10_avg = numpy.mean(djfirst10_history_fitnesses, axis = 0)
set_graph("SOMA de jong first, d = 10, avg", 0, 5000, 0, 50)
plt.plot(range(1, len(djfirst10_avg) + 1), djfirst10_avg, linewidth = 1)
plt.savefig("djfirst_10_avg.png")
plt.clf()
plt.cla()
plt.close()


# run de jong first, d = 30
djfirst30_best_fitnesses, djfirst30_history_fitnesses = run_algorithm(30, -5, 5, de_jong_first)

set_graph("SOMA de jong first, d = 30", 0, 10000, 0, 100)
for hf in djfirst30_history_fitnesses:
    plt.plot(range(1, len(hf) + 1), hf, linewidth = 0.5)
plt.savefig("djfirst_30.png")
plt.clf()
plt.cla()
plt.close()

djfirst30_min = numpy.min(djfirst30_best_fitnesses)
djfirst30_max = numpy.max(djfirst30_best_fitnesses)
djfirst30_mean = numpy.mean(djfirst30_best_fitnesses)
djfirst30_median = numpy.median(djfirst30_best_fitnesses)
djfirst30_std = numpy.std(djfirst30_best_fitnesses)
print(f"djfirst30 min: {djfirst30_min} max: {djfirst30_max} mean: {djfirst30_mean} median: {djfirst30_median} std: {djfirst30_std}")

djfirst30_avg = numpy.mean(djfirst30_history_fitnesses, axis = 0)
set_graph("SOMA de jong first, d = 30, avg", 0, 10000, 0, 100)
plt.plot(range(1, len(djfirst30_avg) + 1), djfirst30_avg, linewidth = 1)
plt.savefig("djfirst_30_avg.png")
plt.clf()
plt.cla()
plt.close()

###

# run de jong second, d = 10
djsecond10_best_fitnesses, djsecond10_history_fitnesses = run_algorithm(10, -2, 2, de_jong_second)

set_graph("SOMA de jong second, d = 10", 0, 50000, 0, 50)
for hf in djsecond10_history_fitnesses:
    plt.plot(range(1, len(hf) + 1), hf, linewidth = 0.5)
plt.savefig("djsecond_10.png")
plt.clf()
plt.cla()
plt.close()

djsecond10_min = numpy.min(djsecond10_best_fitnesses)
djsecond10_max = numpy.max(djsecond10_best_fitnesses)
djsecond10_mean = numpy.mean(djsecond10_best_fitnesses)
djsecond10_median = numpy.median(djsecond10_best_fitnesses)
djsecond10_std = numpy.std(djsecond10_best_fitnesses)
print(f"djsecond10 min: {djsecond10_min} max: {djsecond10_max} mean: {djsecond10_mean} median: {djsecond10_median} std: {djsecond10_std}")

djsecond10_avg = numpy.mean(djsecond10_history_fitnesses, axis = 0)
set_graph("SOMA de jong second, d = 10, avg", 0, 50000, 0, 50)
plt.plot(range(1, len(djsecond10_avg) + 1), djsecond10_avg, linewidth = 1)
plt.savefig("djsecond_10_avg.png")
plt.clf()
plt.cla()
plt.close()


# run de jong second, d = 30
djsecond30_best_fitnesses, djsecond30_history_fitnesses = run_algorithm(30, -2, 2, de_jong_second)

set_graph("SOMA de jong second, d = 30", 0, 5000, 0, 5000)
for hf in djsecond30_history_fitnesses:
    plt.plot(range(1, len(hf) + 1), hf, linewidth = 0.5)
plt.savefig("djsecond_30.png")
plt.clf()
plt.cla()
plt.close()

djsecond30_min = numpy.min(djsecond30_best_fitnesses)
djsecond30_max = numpy.max(djsecond30_best_fitnesses)
djsecond30_mean = numpy.mean(djsecond30_best_fitnesses)
djsecond30_median = numpy.median(djsecond30_best_fitnesses)
djsecond30_std = numpy.std(djsecond30_best_fitnesses)
print(f"djsecond30 min: {djsecond30_min} max: {djsecond30_max} mean: {djsecond30_mean} median: {djsecond30_median} std: {djsecond30_std}")

djsecond30_avg = numpy.mean(djsecond30_history_fitnesses, axis = 0)
set_graph("SOMA de jong second, d = 30, avg", 0, 5000, 0, 5000)
plt.plot(range(1, len(djsecond30_avg) + 1), djsecond30_avg, linewidth = 1)
plt.savefig("djsecond_30_avg.png")
plt.clf()
plt.cla()
plt.close()

###

# run rastrigin, d = 10
rastrigin10_best_fitnesses, rastrigin10_history_fitnesses = run_algorithm(10, -2, 2, rastrigin)

set_graph("SOMA rastrigin, d = 10", 0, 500, -500, 500)
for hf in rastrigin10_history_fitnesses:
    plt.plot(range(1, len(hf) + 1), hf, linewidth = 0.5)
plt.savefig("rastrigin_10.png")
plt.clf()
plt.cla()
plt.close()

rastrigin10_min = numpy.min(rastrigin10_best_fitnesses)
rastrigin10_max = numpy.max(rastrigin10_best_fitnesses)
rastrigin10_mean = numpy.mean(rastrigin10_best_fitnesses)
rastrigin10_median = numpy.median(rastrigin10_best_fitnesses)
rastrigin10_std = numpy.std(rastrigin10_best_fitnesses)
print(f"rastrigin10 min: {rastrigin10_min} max: {rastrigin10_max} mean: {rastrigin10_mean} median: {rastrigin10_median} std: {rastrigin10_std}")

rastrigin10_avg = numpy.mean(rastrigin10_history_fitnesses, axis = 0)
set_graph("SOMA rastrigin, d = 10, avg", 0, 500, -500, 500)
plt.plot(range(1, len(rastrigin10_avg) + 1), rastrigin10_avg, linewidth = 1)
plt.savefig("rastrigin_10_avg.png")
plt.clf()
plt.cla()
plt.close()


# run rastrigin, d = 30
rastrigin30_best_fitnesses, rastrigin30_history_fitnesses = run_algorithm(30, -2, 2, rastrigin)

set_graph("SOMA rastrigin, d = 30", 0, 500, -500, 500)
for hf in rastrigin30_history_fitnesses:
    plt.plot(range(1, len(hf) + 1), hf, linewidth = 0.5)
plt.savefig("rastrigin_30.png")
plt.clf()
plt.cla()
plt.close()

rastrigin30_min = numpy.min(rastrigin30_best_fitnesses)
rastrigin30_max = numpy.max(rastrigin30_best_fitnesses)
rastrigin30_mean = numpy.mean(rastrigin30_best_fitnesses)
rastrigin30_median = numpy.median(rastrigin30_best_fitnesses)
rastrigin30_std = numpy.std(rastrigin30_best_fitnesses)
print(f"rastrigin30 min: {rastrigin30_min} max: {rastrigin30_max} mean: {rastrigin30_mean} median: {rastrigin30_median} std: {rastrigin30_std}")

rastrigin30_avg = numpy.mean(rastrigin30_history_fitnesses, axis = 0)
set_graph("SOMA rastrigin, d = 30, avg", 0, 500, -500, 500)
plt.plot(range(1, len(rastrigin30_avg) + 1), rastrigin30_avg, linewidth = 1)
plt.savefig("rastrigin_30_avg.png")
plt.clf()
plt.cla()
plt.close()

###

# run schwefel, d = 10
schwefel10_best_fitnesses, schwefel10_history_fitnesses = run_algorithm(10, 0, 500, schwefel)

set_graph("SOMA schwefel, d = 10", 0, 50000, -5000, 0)
for hf in schwefel10_history_fitnesses:
    plt.plot(range(1, len(hf) + 1), hf, linewidth = 0.5)
plt.savefig("schwefel_10.png")
plt.clf()
plt.cla()
plt.close()

schwefel10_min = numpy.min(schwefel10_best_fitnesses)
schwefel10_max = numpy.max(schwefel10_best_fitnesses)
schwefel10_mean = numpy.mean(schwefel10_best_fitnesses)
schwefel10_median = numpy.median(schwefel10_best_fitnesses)
schwefel10_std = numpy.std(schwefel10_best_fitnesses)
print(f"schwefel10 min: {schwefel10_min} max: {schwefel10_max} mean: {schwefel10_mean} median: {schwefel10_median} std: {schwefel10_std}")

schwefel10_avg = numpy.mean(schwefel10_history_fitnesses, axis = 0)
set_graph("SOMA schwefel, d = 10, avg", 0, 50000, -5000, 0)
plt.plot(range(1, len(schwefel10_avg) + 1), schwefel10_avg, linewidth = 1)
plt.savefig("schwefel_10_avg.png")
plt.clf()
plt.cla()
plt.close()


# run schwefel, d = 30
schwefel30_best_fitnesses, schwefel30_history_fitnesses = run_algorithm(30, 0, 500, schwefel)

set_graph("SOMA schwefel, d = 30", 0, 1000, -5000, 0)
for hf in schwefel30_history_fitnesses:
    plt.plot(range(1, len(hf) + 1), hf, linewidth = 0.5)
plt.savefig("schwefel_30.png")
plt.clf()
plt.cla()
plt.close()

schwefel30_min = numpy.min(schwefel30_best_fitnesses)
schwefel30_max = numpy.max(schwefel30_best_fitnesses)
schwefel30_mean = numpy.mean(schwefel30_best_fitnesses)
schwefel30_median = numpy.median(schwefel30_best_fitnesses)
schwefel30_std = numpy.std(schwefel30_best_fitnesses)
print(f"schwefel30 min: {schwefel30_min} max: {schwefel30_max} mean: {schwefel30_mean} median: {schwefel30_median} std: {schwefel30_std}")

schwefel30_avg = numpy.mean(schwefel30_history_fitnesses, axis = 0)
set_graph("SOMA schwefel, d = 30, avg", 0, 1000, -5000, 0)
plt.plot(range(1, len(schwefel30_avg) + 1), schwefel30_avg, linewidth = 1)
plt.savefig("schwefel_30_avg.png")
plt.clf()
plt.cla()
plt.close()
