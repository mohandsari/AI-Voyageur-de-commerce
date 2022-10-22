import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import time


# minialspanningtree
def mst(cities_in_tree, cities_outside, mat):
    minimal_path = 0
    cities_in_tree = cities_in_tree
    cities_outside = cities_outside
    while len(cities_outside) != 0:
        # look for the closest city
        ll = list(itertools.product(cities_in_tree, cities_outside))
        all_dist = [mat[s[0]][s[1]] for s in ll]
        index_min = np.argmin(all_dist)
        closest_city = ll[index_min][1]
        minimal_path = minimal_path + min(all_dist)
        # city is added to the graph
        cities_in_tree = np.append(cities_in_tree, closest_city)
        cities_outside = cities_outside[cities_outside != closest_city]
    return minimal_path


def path_cost(path, matrix):  # given an complete hamiltonaian circus compute the cost of the circus
    n = len(path)
    if n == 1:
        cost = [0]
    else:
        cost = map(lambda x: matrix[path[x]][path[x+1]], np.arange(n-1))
    return sum(cost)


def a_star(init_city, matrix):
    nb_iteration = 0
    start_time = time.time()
    print("A STAR ALGORITHM *")
    cost = 0
    closed_list = np.array([init_city])  # already visited cities
    open_list = matrix.index[matrix.index != init_city] # not yet visited cities
    current_node = init_city
    frontier = []
    max_frontier = 0
    while len(open_list) > 1:
        # look for the best node among the non visited cities
        # f = g + h
        # to evaluate h: compute the minimal spanning tree with unvisited cities
        # and add the value of the min edge which connect this tree to the current city and final city
        min_node = 0
        f_min = np.infty
        for s in open_list:
            unvisited = open_list[open_list != s]
            mst_unvisited = mst(np.array([unvisited[0]]), unvisited, matrix)
            h_s = mst(unvisited, np.array([s]), matrix) + mst(unvisited, np.array([init_city]), matrix) + mst_unvisited
            g_s = path_cost(closed_list, matrix) + matrix[current_node][s]  # for g compute edges from initial+last edge
            f_s = h_s + g_s
            if f_s < f_min:
                f_min = f_s
                min_node = s
            frontier.append([s, f_s])
            print("Node: ", s, "f: ", f_s, "h: ", h_s, "g: ", g_s)
        print("frontier: ", frontier)
        frontier.remove([min_node, f_min])  # remove the chosen node
        # update cost
        cost = cost + matrix[current_node][min_node]
        # add him to the closed list
        # remove him from the open list
        closed_list = np.append(closed_list, min_node)
        open_list = open_list[open_list != min_node]
        current_node = min_node
        if len(frontier) > max_frontier:
            max_frontier = len(frontier)
        print("Explored cities: ", closed_list)
        nb_iteration = nb_iteration + 1
    # cost to return to the initial position
    # only one city left to visit: no choice
    cost = cost + matrix[current_node][open_list[0]] + matrix[init_city][open_list[0]]
    circuit = np.concatenate((closed_list, [open_list[0], init_city]))
    print("TIME FOR A* STAR --- %s seconds ---" % (time.time() - start_time))
    total_time = int(time.time() - start_time)
    return circuit, cost, max_frontier, total_time, nb_iteration


def two_opt(path):  # take a circus and change two edge, circus need to have length >= 4 and return new circus
    n_path = path.copy()
    n = len(path)
    ind = np.delete(np.arange(n), (0, -2, -1))  # initial and final city can not be changed
    i = np.random.choice(ind)
    j = np.random.choice(np.delete(np.arange(n), (0, i-1, i, i+1, -1)))
    tmp = n_path[i+1]
    n_path[i+1] = n_path[j]
    n_path[j] = tmp
    return n_path


def hill_climbing(init_city, matrix, it):
    start_time = time.time()
    current_path = np.array(matrix.index[matrix.index != init_city])  # all cities without initial
    np.random.shuffle(current_path)  # random circuit
    current_path = np.concatenate(([init_city], current_path, [init_city]))  # hamiltonian circuit
    init_cost = path_cost(current_path, matrix)
    i = 0
    while i < it:
        new_path = two_opt(path=current_path)
        print(new_path)
        delta_cost = path_cost(new_path, matrix) - path_cost(current_path, matrix)
        # if new circuit is better, take it
        if delta_cost < 0:
            print("Amelioration: ", -delta_cost)
            current_path = new_path
        else:
            print("No Amelioration")
        i = i + 1
    final_cost = path_cost(current_path, matrix)
    to = int((final_cost/init_cost)*100)
    print("TIME FOR HILL CLIMBING --- %s seconds ---" % (time.time() - start_time))
    total_time = int(time.time() - start_time)
    return current_path, final_cost, to, total_time, it


def generate_graph(nb_cities, max_distance):
    n = nb_cities
    distances = np.random.choice(np.arange(max_distance), n*n).reshape((n, n))
    m = np.tril(distances) + np.tril(distances, -1).T
    np.fill_diagonal(m, np.zeros(n))
    data = pd.DataFrame(m)
    return data

# 1er étape : Compiler tous ce qu'il y a au dessus ( fonctions qu'on utilise )
# ############# RUN THE CODE ############


# graph sample
dist_matrix = np.array([[0, 21, 9, 21, 1],
                        [21, 0, 4, 8, 20],
                        [9, 4, 0, 6, 7],
                        [21, 8, 6, 0, 11],
                        [1, 20, 7, 11, 0]])

graph_1 = pd.DataFrame([[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]])
graph_2 = pd.DataFrame([[0,300,200],[300,0,500],[200,500,0]])
dd = pd.DataFrame(dist_matrix)

# graph generation with 3 cities
gg = generate_graph(nb_cities=3, max_distance=30)
# ####### A STAR ################
ans_a = a_star(init_city=0, matrix=gg)
opt_path = ans_a[0]
opt_cost = ans_a[1]
print("****  RESULT FOR A* ALGORITHM ****")
print("Optimal path: ", opt_path)
print("Optimal cost is: ", opt_cost)


#partie 2 : compiler A* de cette ligne jusqu'à la ligne suivant la 1ère étape
# ########## Hill climbing and comparison to A* ##############
matrix = generate_graph(nb_cities=10, max_distance=25)
init_city=0
it=1000
ans = hill_climbing(init_city=0, matrix=matrix, it=1000)
h_path = ans[0]
h_cost = ans[1]
h_to = ans[2]
print("****  RESULT FOR HILL CLIMBING ****")
print("Final path: ", h_path)
print("Cost is: ", h_cost)
print("Amelioration % from initial circuit: ", h_to, "%")
print("Distance to optimum:", h_cost-opt_cost)

# partie 3: compiler Hill Climbing de cette ligne jusqu'à la ligne suivant la partie 2 

# Calibrage, Itération de Hill Climbing en fonction du coût minimum trouvé
# prend environ une minute à la compilation
#On compile le code en dessous(le calibrage)
x_range = np.arange(500)
y_range = [hill_climbing(0, matrix, x)[1] for x in x_range]
plt.plot(x_range, y_range, label="hill climbing")
plt.xlabel('iteration')
plt.ylabel('cout')
plt.legend(loc="upper left")
plt.show()

# TABLE A*
#On compile le code en dessous
MAX_VILLE = 25
MIN_VILLE = 6
graph = generate_graph(MAX_VILLE, 20)
res = [a_star(init_city=0, matrix=graph.iloc[0:n, 0:n])
       for n in np.arange(MIN_VILLE, MAX_VILLE)]
table = pd.DataFrame(data=res)
table.insert(0, 'Nombre de ville', np.arange(MIN_VILLE, MAX_VILLE))
table = table.drop(0, 1)
table.columns = ['Nb de ville', 'Coût', 'Frontier max', 'Temps', 'Nb Iteration']
print(table)


# TABLE Hill Climbing
# hill climbing can not run if nb ville < 5
#On compile le code en dessous

graph = generate_graph(MAX_VILLE, 20)
res = [hill_climbing(init_city=0, matrix=graph.iloc[0:n, 0:n], it=100)
       for n in np.arange(MIN_VILLE, MAX_VILLE)]
table = pd.DataFrame(data=res)
table.insert(0, 'Nombre de ville', np.arange(MIN_VILLE, MAX_VILLE))
table = table.drop(0, 1)
table.columns = ['Nb de ville', 'Coût', 'Amelioration %', 'Temps', 'Nb Iteration']
print(table)

