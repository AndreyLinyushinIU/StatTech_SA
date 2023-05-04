import numpy as np
import numpy.random
from haversine import haversine, Unit
from matplotlib import pyplot as plt
import csv
import imageio.v2 as imageio

class SimulatedAnnealing:
    def __init__(self, annealing_rate, temp_initial, temp_cooldown):
        self.temp_cooldown = temp_cooldown
        self.nodes = {}
        self.n_cities = 30
        with open('cities.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                idx = reader.line_num-1
                self.nodes[idx] = [row[0], float(row[1]), float(row[2])]
                if idx == self.n_cities - 1:
                    break
        self.i = 0
        self.annealing_rate = annealing_rate
        self.temp = temp_initial
        self.path = np.array([*self.nodes.keys()])
        self.cost = self.eval_cost(self.path)

    def anneal(self):
        res = []
        while self.temp >= self.temp_cooldown:
            # generating proposal path
            idx = np.random.choice(len(self.path), size=2)
            proposal = self.path.copy()
            proposal[idx[0]], proposal[idx[1]] = proposal[idx[1]], proposal[idx[0]]
            proposal_cost = self.eval_cost(proposal)
            # sampling from uniform distribution
            u = np.random.uniform()
            p_star = np.exp(-proposal_cost / self.temp)
            # accepting or remaining the same
            if proposal_cost < self.cost or u < p_star:
                self.path = proposal
                self.cost = proposal_cost
            self.temp *= self.annealing_rate
            self.i += 1
            res.append(np.concatenate(([self.i, self.cost], self.path)))
        return np.array(res)

    def eval_cost(self, path):
        return sum([haversine(self.nodes[path[i]][1:], self.nodes[path[i + 1]][1:], unit=Unit.KILOMETERS) for i in range(len(self.path) - 1)])



annealing = SimulatedAnnealing(annealing_rate=0.99, temp_initial=100000, temp_cooldown=0.0000001)
MAP_WIDTH = 500
MAP_HEIGHT = 500
#x = [int((MAP_WIDTH / 360.0) * (180 + annealing.nodes[i][1])) for i in range(annealing.n_cities)]
#y = [int((MAP_HEIGHT / 180.0) * (90 - annealing.nodes[i][2])) for i in range(annealing.n_cities)]
y = [annealing.nodes[i][1] for i in range(annealing.n_cities)]
x = [annealing.nodes[i][2] for i in range(annealing.n_cities)]
names = [annealing.nodes[i][0] for i in range(annealing.n_cities)]
res = annealing.anneal()

for i in range(len(res)):
    print(str(i/len(res)*100)+'%')
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    j = 0
    for name in names:
        plt.scatter(x[j], y[j], marker='.', color='red')
        plt.text(x[j] + .03, y[j] + .03, name, fontsize=9)
        j += 1

    path_numbers = res[i][2:]
    path_coords = np.array([[x[int(n)], y[int(n)]] for n in path_numbers]).T
    ax.plot(path_coords[0], path_coords[1], color='black')
    fig.savefig('images/'+str(i)+'.png', bbox_inches='tight')
    plt.close()


images = []
for filename in ['images/'+str(i)+'.png' for i in range(len(res))]:
    images.append(imageio.imread(filename))
imageio.mimsave('annealing.gif', images)