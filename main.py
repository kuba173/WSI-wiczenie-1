import random
import numpy as np
import pygame
import copy
n_cities = 30
n_population = 30
n_iterations = 400
tournament_size = 2
mutation_rate = 0.2
generation = True
the_best_path = []
best_path=[]
population = []
#cities = np.random.rand(n_cities, 2)

cities = np.array([[0.15279738, 0.11679128],
 [0.93067568, 0.01298152],
 [0.87676058, 0.31273269],
 [0.44704925, 0.0628451],
 [0.10394095, 0.2125702],
 [0.0730121, 0.11701134],
 [0.17740486, 0.98108268],
 [0.45518775, 0.02398185],
 [0.56961308, 0.63315214],
 [0.53868077, 0.13937475],
 [0.31031544, 0.34561839],
 [0.95107601, 0.3933266],
 [0.07443927, 0.8559512],
 [0.46005358, 0.39474773],
 [0.76032808, 0.58449763],
 [0.10571585, 0.43954825],
 [0.5308176, 0.55780057],
 [0.40451671, 0.10923664],
 [0.03089046, 0.22927667],
 [0.16015457, 0.38539105],
 [0.21150104, 0.74873412],
 [0.78221031, 0.76804445],
 [0.11509798, 0.57895287],
 [0.84919854, 0.8685739],
 [0.52816538, 0.87100725],
 [0.83416445, 0.8351582],
 [0.31339992, 0.0581748],
 [0.99442961, 0.34229637],
 [0.38377987, 0.73723087],
 [0.94400944, 0.10063084],
[0.15279738, 0.11679128]])


print("cities",cities)

def mutation(path, mutation_rate):

    for i in range(3):

        if np.random.rand() < mutation_rate:

            idx1, idx2 = np.random.choice(len(path)-2, size=2, replace=False)
            idx1 = idx1 + 1
            idx2 = idx2 + 1

            path[idx1], path[idx2] = copy.deepcopy(path[idx2]), copy.deepcopy(path[idx1])
    return path


def selection(population):

    selected = []
    winner = min(population, key=total_distance)


    for i in range(len(population)):
        selected.append(winner)

    return selected

def tournament_selection(population, k):

    selected = []
    for i in range(len(population)):
        tournament = random.sample(population, k)

        winner = min(tournament, key=lambda x: total_distance(x))  # choosing the winner

        selected.append(winner)


    return selected

def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)


def total_distance(path):
    dist = 0
    for i in range(len(path)-1):

        dist += distance(cities[path[i]], cities[path[i+1]])
    dist += distance(cities[path[-1]], cities[path[0]])
    return dist

for i in range(n_population):
    permutation = np.random.permutation(np.arange(1, n_cities))
    permutation = np.insert(permutation, 0, 0)  # adding 0 on beginning
    permutation = np.append(permutation, [0])  # adding 0 on back
    population.append(permutation)

print("population ",population)

offspring = copy.deepcopy(population)
for it in range(n_iterations):

    offspring[len(offspring) - 1] = copy.deepcopy(min(population, key=total_distance))

    selected = tournament_selection(population, tournament_size)

    selected[len(selected) - 1] = min(offspring, key=total_distance)
    offspring = copy.deepcopy(selected)

    for i in range(len(selected) - 2):
        offspring[i + 1] = copy.deepcopy(mutation(selected[i + 1], mutation_rate))

    population = copy.deepcopy(offspring)

    best_path = min(population, key=total_distance)
    print("Iteration:", it, "best path:", best_path, "distanc lenght:", total_distance(best_path))
    if it == 0:
        the_best_path = best_path

    if total_distance(best_path) < total_distance(the_best_path):
        the_best_path = best_path.copy()

print("Iteration:", the_best_path, "The best path:", total_distance(the_best_path))


##################################
pygame.init()

window = pygame.display.set_mode((300, 300))
window.fill((255, 255, 255))
run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    pygame.draw.circle(window, (255, 0, 0), (cities[the_best_path[0]][0] * 300, cities[the_best_path[0]][1] * 300), 12)
    for i in range(len(the_best_path)):

        distance(cities[the_best_path[i-1]], cities[the_best_path[i]])
        pygame.draw.circle(window, (0, 0, 0), (cities[the_best_path[i]][0]*300,cities[the_best_path[i]][1]*300), 4)


        if i>0:
            pygame.draw.line(window, (0, 0, 0), (cities[the_best_path[i-1]][0]*300,cities[the_best_path[i-1]][1]*300), (cities[the_best_path[i]][0]*300,cities[the_best_path[i]][1]*300))

    pygame.display.flip()

pygame.quit()

