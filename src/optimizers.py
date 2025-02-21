import numpy as np
from typing import Tuple, List, Callable
import random
import warnings

class GeneticOptimizer:
    def __init__(self,
                 fitness_func: Callable,
                 population_size: int = 100,
                 n_generations: int = 1000,
                 mutation_rate: float = 0.15,
                 elite_size: int = 10):
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.current_generation = 0

        # Parameter bounds
        self.bounds = {
            'b': (0.001, 1),  # Magnetic field strength (avoiding zero)
            'theta': (0.001, np.pi-0.0001),  # Polar angle
            'phi': (0, 2 * np.pi),  # Azimuthal angle
            'exg': (0, 1e-3),  # Strain parameters
            'exyg': (0, 1e-3),
        }

    def create_individual(self) -> dict:
        """Create a random individual within bounds"""
        return {
            param: random.uniform(bounds[0], bounds[1])
            for param, bounds in self.bounds.items()
        }

    def create_initial_population(self) -> List[dict]:
        """Create initial random population with diverse values"""
        population = []

        # Add some individuals at the bounds
        for _ in range(20):
            individual = {}
            for param, bounds in self.bounds.items():
                if random.random() < 0.5:
                    individual[param] = bounds[0]
                else:
                    individual[param] = bounds[1]
            population.append(individual)

        # Fill rest with random individuals
        while len(population) < self.population_size:
            population.append(self.create_individual())

        return population

    def evaluate_fitness(self, individual: dict) -> float:
        """Evaluate the fitness of an individual - lower is better"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.fitness_func(
                    individual['b'],
                    individual['theta'],
                    individual['phi'],
                    individual['exg'],
                    individual['exyg']
                )
            # Check for NaN or inf
            if np.isnan(result) or np.isinf(result):
                return float('inf')
            return result
        except:
            return float('inf')

    def select_parents(self, population: List[dict], fitnesses: List[float]) -> Tuple[dict, dict]:
        """Select parents using tournament selection - lower fitness is better"""
        tournament_size = 5

        def tournament():
            indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in indices]
            winner_idx = indices[np.argmin(tournament_fitnesses)]
            return population[winner_idx]

        return tournament(), tournament()

    def crossover(self, parent1: dict, parent2: dict) -> dict:
        """Perform arithmetic crossover between parents"""
        child = {}
        for param in self.bounds.keys():  # Fixed: iterate over parameter names
            weight = random.random()
            child[param] = weight * parent1[param] + (1 - weight) * parent2[param]
        return child

    def mutate(self, individual: dict) -> dict:
        """Mutate an individual with adaptive mutation rate"""
        mutated = individual.copy()
        for param, bounds in self.bounds.items():
            if random.random() < self.mutation_rate:
                param_range = bounds[1] - bounds[0]
                # Adaptive mutation: larger changes early, smaller changes later
                scale = max(0.01, 1 - (self.current_generation / self.n_generations))
                mutation = random.uniform(-0.2, 0.2) * param_range * scale
                mutated[param] = np.clip(
                    individual[param] + mutation,
                    bounds[0],
                    bounds[1]
                )
        return mutated

    def optimize(self) -> Tuple[dict, float]:
        """Run the genetic algorithm optimization"""
        population = self.create_initial_population()

        best_individual = None
        best_fitness = float('inf')
        generations_without_improvement = 0

        for generation in range(self.n_generations):
            self.current_generation = generation

            # Evaluate fitness for all individuals
            fitnesses = [self.evaluate_fitness(ind) for ind in population]

            # Update best solution
            min_fitness_idx = np.argmin(fitnesses)
            if fitnesses[min_fitness_idx] < best_fitness:
                best_fitness = fitnesses[min_fitness_idx]
                best_individual = population[min_fitness_idx].copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Sort population by fitness (ascending)
            sorted_indices = np.argsort(fitnesses)
            population = [population[i] for i in sorted_indices]

            # Create new population
            new_population = []

            # Keep elite individuals
            new_population.extend(population[:self.elite_size])

            # Create rest of new population
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population, fitnesses)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

            # Print progress
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness}")

            # Early stopping if we're very close to zero
            if best_fitness < 1e-11:
                print(f"Found solution very close to zero at generation {generation}")
                break

            # Reset mutation rate if stuck
            if generations_without_improvement > 20:
                self.mutation_rate = min(0.3, self.mutation_rate * 1.5)
                generations_without_improvement = 0
                print(f"Increasing mutation rate to {self.mutation_rate}")

        return best_individual, best_fitness



def optimize_function(f, otype = 'Genetic'):
    if otype == 'Genetic':
        optimizer = GeneticOptimizer(
            fitness_func=f,
            population_size=50,
            n_generations=1000,
            mutation_rate=0.15,
            elite_size=15
        )


    best_solution, best_fitness = optimizer.optimize()

    print("\nOptimization completed!")
    print("Best solution found:")
    for param, value in best_solution.items():
        print(f"{param}= {value:.6f}")
    print(f"Final dot product value: {best_fitness:.8f}")

    return best_solution, best_fitness