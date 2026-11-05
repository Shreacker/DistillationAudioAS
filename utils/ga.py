import numpy as np

def fitness_fn(weights, val_preds, y_true):
    blended = np.tensordot(weights, val_preds, axes=1)
    blended = np.clip(blended, 1e-12, 1.0)

    return float(np.mean(np.log(blended[np.arange(len(y_true)), y_true])))

def normalize(w, eps=1e-12):
    w = np.clip(w, 0.0, None)
    s = w.sum(axis=-1, keepdims=True)
    w = np.where(s < eps, 1.0 / w.shape[-1], w / np.maximum(s, eps))
    
    return w

def tournament_select(pop, fit, k, rng):
    idx = rng.integers(0, len(pop), size=k)
    
    return pop[idx[np.argmax(fit[idx])]]

def crossover(p1, p2, rng):
    a = rng.random()

    return a*p1 + (1.0 - a)*p2

def mutate(w, rate, sigma, rng):
    mask = rng.random(w.shape) < rate
    w = w + mask*rng.normal(0.0, sigma, size=w.shape)

    return normalize(w)

class EnsembleWeightGA:
    def __init__(
            self,
            n_models=5,
            pop_size=50,
            elite=2,
            tournament_k=3,
            crossover_rate=0.9,
            mutation_rate=0.2,
            mutation_sigma=0.1,
            sigma_decay=0.98,
            seed=21
    ):
        self.M = n_models
        self.pop_size = pop_size
        self.elite = elite
        self.k = tournament_k
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.sigma = mutation_sigma
        self.sigma_decay = sigma_decay
        self.rng = np.random.default_rng(seed)
        
        self.pop = self._init_population()
        self.best_w = np.full(n_models, 1.0 / n_models)
        self.best_fit = -np.inf

    def _init_population(self):
        pop = self.rng.dirichlet(np.ones(self.M), size=self.pop_size)
        pop[0] = np.full(self.M, 1.0 / self.M)
        
        for i in range(1, min(self.M + 1, self.pop_size)):
            pop[i] = normalize(np.eye(self.M)[i - 1] * 0.8 + 0.04)

        return pop

    def evolve(self, val_preds, y_true, generations=30, patience=10, verbose=False):
        best_gen, stall = -np.inf, 0

        for gen in range(generations):
            fit = np.array([fitness_fn(w, val_preds, y_true) for w in self.pop])

            i_best = int(np.argmax(fit))
            if fit[i_best] > self.best_fit:
                self.best_fit = fit[i_best]
                self.best_w = self.pop[i_best].copy()

            if fit[i_best] > best_gen + 1e-9:
                best_gen, stall = fit[i_best], 0
            else:
                stall += 1
            if verbose:
                print(f" gen {gen:3d} | best {fit[i_best]:.5f} | mean {fit.mean():.5f}")
            if stall >= patience:
                break

            order = np.argsort(fit)[::-1]
            new_pop = [self.pop[i].copy() for i in order[:self.elite]]

            while len(new_pop) < self.pop_size:
                p1 = tournament_select(self.pop, fit, self.k, self.rng)
                p2 = tournament_select(self.pop, fit, self.k, self.rng)
                child = crossover(p1, p2, self.rng) if self.rng.random() < self.cx_rate else p1.copy()
                child = mutate(child, self.mut_rate, self.sigma, self.rng)
                new_pop.append(child)

            self.pop = np.array(new_pop)
            self.sigma *= self.sigma_decay

        self.best_w = normalize(self.best_w)
        
        return self.best_w