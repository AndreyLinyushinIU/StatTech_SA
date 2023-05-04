import numpy as np
import numpy.random
from haversine import haversine, Unit
from matplotlib import pyplot as plt
import csv

class SimulatedAnnealing:
    def __init__(self, annealing_rate, temp_initial, temp_cooldown):
        self.temp_cooldown = temp_cooldown
        self.nodes = {}
        n_cities = 30
        with open('cities.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                idx = reader.line_num-1
                self.nodes[idx] = [row[0], float(row[1]), float(row[2])]
                if idx == n_cities - 1:
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

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 1, 1)
rates = [0.5, 0.7, 0.9, 0.95, 0.99]
generate_gif = True

for i in range(len(rates)):
    annealing = SimulatedAnnealing(annealing_rate=rates[i], temp_initial=100000, temp_cooldown=0.0000001)
    res = annealing.anneal()
    if i == len(rates) - 1 and generate_gif:
        path = res[-1][2:]
        print(len(path), path)

    ax.plot(res.T[0][:], res.T[1][:], label=fr'$T_0=10^5$, $T_f=0.1^7$, rate={rates[i]}, iter={len(res)}')
    ax.grid(color='black', linestyle='--', linewidth=1.0, alpha = 0.7)
    ax.grid(True)
    ax.legend()
    ax.set_ylabel(r'Distance, $km$')
    ax.set_xlabel(r'Iterations')
    ax.ticklabel_format(style='plain')
    fig.savefig('foo.png', bbox_inches='tight')
plt.show()