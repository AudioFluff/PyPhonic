import random

import numpy as np
from scipy.io.wavfile import write
import librosa

from pyphonic.preset_19_synth import Poly, Synth, get_preset

MAX_PARAMS = 9

# poly = Poly(**get_preset("basic"))
poly = Poly(stack=[Synth(op="sin", detune_coarse=0, detune=1, rel_vel=0.5, phase=20, attack=0.1, decay=0.1, sustain=0.5, release=30),
                   Synth(op="tri", detune_coarse=0, detune=0, rel_vel=0.5, phase=0),
                   Synth(op="noise", detune_coarse=0, detune=0, rel_vel=0.1, phase=0),
                   ])
# print(poly.synth_stack, poly.synths)

poly.set_sample_rate_block_size(44100, 256)
for i, filter in enumerate(poly.filters):
    filter.set_sample_rate_block_size(44100, 256, i)
for i, lfo in enumerate(poly.lfos):
    lfo_synth, freq, vel = lfo
    lfo_synth.set_sample_rate_block_size(44100, 256, i)
    lfo_synth.start_freq(freq, vel)

poly.start_note(60, 10)

ga_input = np.array([])

for _ in range(200):
    ga_input = np.concatenate((ga_input, poly.render()))
poly.stop_note(60)
for _ in range(200):
    ga_input = np.concatenate((ga_input, poly.render()))

scaled = np.int16(ga_input / np.max(np.abs(ga_input)) * 32767)
write('/tmp/ga_input.wav', 44100, scaled)
ga_input = ga_input / np.max(np.abs(ga_input))

def encode_parameters(num_oscillators, oscillators):
    # Assuming osc_type is already an integer
    params = [num_oscillators]
    for i in range(num_oscillators):
        osc_type, detune_coarse, detune, rel_vel, phase, attack, decay, sustain, release = oscillators[i]
        params.extend([osc_type, detune_coarse, detune, rel_vel, phase, attack, decay, sustain, release])
    return params

def decode_parameters(chromosome):
    num_oscillators = chromosome[0]
    ops = {
        0: "square",
        1: "saw",
        2: "tri",
        3: "sin",
        4: "noise"
    }
    stack = []
    for i in range(1, len(chromosome), MAX_PARAMS):
        synth = {}
        synth["op"] = ops[chromosome[i]]
        synth["detune_coarse"] = chromosome[i + 1]
        synth["detune"] = chromosome[i + 2]
        synth["rel_vel"] = chromosome[i + 3]
        synth["phase"] = chromosome[i + 4]
        synth["attack"] = chromosome[i + 5]
        synth["decay"] = chromosome[i + 6]
        synth["sustain"] = chromosome[i + 7]
        synth["release"] = chromosome[i + 8]
        stack.append((Synth, synth))
    return stack

def initialize_population(size):
    population = []
    for _ in range(size):
        num_oscillators = np.random.randint(1, 5)  # 1-4 oscillators
        oscillators = []
        for _ in range(num_oscillators):
            osc_type = np.random.randint(0, 5)  # square, saw, tri, sin, noise
            detune_coarse = int(np.random.triangular(-12, 0, 12))
            detune_fine = int(np.random.triangular(-100, 0, 101))
            rel_vel = np.round(np.random.uniform(0.1, 1.0), 2)
            phase = int(np.random.triangular(-360, 0, 361))
            attack = np.round(np.random.triangular(0, 0, 1), 2)
            decay = np.round(np.random.triangular(0, 0, 1), 2)
            sustain = np.round(np.random.triangular(0, 0.5, 1), 2)
            release = np.round(np.random.triangular(0, 0, 4), 2)
            oscillators.append([osc_type, detune_coarse, detune_fine, rel_vel, phase, attack, decay, sustain, release])
        chromosome = encode_parameters(num_oscillators, oscillators)
        assert len(chromosome) == 1 + (num_oscillators * MAX_PARAMS)
        population.append(chromosome)
    return population

def mutate(chromosome):
    mutation_point = np.random.randint(0, len(chromosome))
    if mutation_point == 0:  # Number of oscillators
        values = np.array([1, 2, 3, 4])
        weights = np.array([0.5, 0.3, 0.15, 0.05])
        weights /= weights.sum()
        value = np.random.choice(values, p=weights)
        if value > chromosome[mutation_point]: ## Need to add params for another osc
            for _ in range(value - chromosome[mutation_point]):
                osc_type = np.random.randint(0, 5)
                detune_coarse = int(np.random.triangular(-12, 0, 12))
                detune_fine = int(np.random.triangular(-100, 0, 101))
                rel_vel = np.round(np.random.uniform(0.1, 1.0), 2)
                phase = int(np.random.triangular(-360, 0, 361))
                attack = np.round(np.random.triangular(0, 0, 1), 2)
                decay = np.round(np.random.triangular(0, 0, 1), 2)
                sustain = np.round(np.random.triangular(0, 0.5, 1), 2)
                release = np.round(np.random.triangular(0, 0, 4), 2)
                chromosome.extend([osc_type, detune_coarse, detune_fine, rel_vel, phase, attack, decay, sustain, release])
        elif value < chromosome[mutation_point]: ## Need to remove params for another osc
            for _ in range(chromosome[mutation_point] - value):
                chromosome = chromosome[:1 + (value * MAX_PARAMS)]
        chromosome[mutation_point] = value
        chromosome = chromosome[:1 + (value * MAX_PARAMS)]
        
    else:  # Oscillator parameters
        
        parameter = (mutation_point - 1) % MAX_PARAMS

        if parameter == 0:  # Oscillator type
            chromosome[mutation_point] = np.random.randint(0, 5)
        elif parameter == 1:  # Detune Coarse
            chromosome[mutation_point] = int(np.random.triangular(-12, 0, 12))  # regularized
        elif parameter == 2:  # Detune Fine
            chromosome[mutation_point] = int(np.random.triangular(-100, 0, 101))
        elif parameter == 3:  # Relative Velocity
            chromosome[mutation_point] = np.round(np.random.uniform(0.1, 1.0), 2)
        elif parameter == 4:  # Phase
            chromosome[mutation_point] = int(np.random.triangular(-360, 0, 361))
        elif parameter == 5:  # Attack
            chromosome[mutation_point] = np.round(np.random.triangular(0, 0, 1), 2)
        elif parameter == 6:  # Decay
            chromosome[mutation_point] = np.round(np.random.triangular(0, 0, 1), 2)
        elif parameter == 7:  # Sustain
            chromosome[mutation_point] = np.round(np.random.triangular(0, 0.5, 1), 2)
        elif parameter == 8:  # Release
            chromosome[mutation_point] = np.round(np.random.triangular(0, 0, 4), 2)
    
    assert len(chromosome) == 1 + (chromosome[0] * MAX_PARAMS)

    return chromosome

def crossover(parent1, parent2):
    parent1 = [x for x in parent1]
    parent2 = [x for x in parent2]
    crossover_point = np.random.randint(0, len(parent1))
    assert len(parent1) == 1 + (parent1[0] * MAX_PARAMS)
    assert len(parent2) == 1 + (parent2[0] * MAX_PARAMS)
    if len(parent1) != len(parent2):
        # 5050 chance to either extend the shorter one with the longer's dna, or cut the longer one
        if random.choice([True, False]):
            # Extend the shorter one
            if len(parent1) < len(parent2):
                parent1[0] = parent2[0]
                parent1.extend(parent2[len(parent1):])
            else:
                parent2[0] = parent1[0]
                parent2.extend(parent1[len(parent2):])
        else:
            # Cut the longer one
            if len(parent1) < len(parent2):
                parent2[0] = parent1[0]
                parent2 = parent2[:len(parent1)]
            else:
                parent1[0] = parent2[0]
                parent1 = parent1[:len(parent2)]
    
    child1 = parent1[:crossover_point] + parent2[crossover_point:]    
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    assert len(child1) == 1 + (child1[0] * MAX_PARAMS)
    assert len(child2) == 1 + (child2[0] * MAX_PARAMS)
    return child1, child2

# population = initialize_population(1)
# p = population[0]

# orig = [x for x in p]
# for _ in range(100):
#     p = mutate(p)
#     decode_parameters(p)

# for _ in range(1000):
#     p = mutate(p)
#     c1, c2 = crossover(orig, p)
#     decode_parameters(c1)
#     decode_parameters(c2)


def simulate(chromosome):
    stack = []
    for item in decode_parameters(chromosome):
        synth, params = item
        stack.append(synth(**params))
    poly = Poly(stack=stack)

    poly.set_sample_rate_block_size(44100, 256)
    for i, filter in enumerate(poly.filters):
        filter.set_sample_rate_block_size(44100, 256, i)
    for i, lfo in enumerate(poly.lfos):
        lfo_synth, freq, vel = lfo
        lfo_synth.set_sample_rate_block_size(44100, 256, i)
        lfo_synth.start_freq(freq, vel)

    poly.start_note(60, 10)

    ga_output = np.array([])

    for _ in range(200):
        ga_output = np.concatenate((ga_output, poly.render()))

    poly.stop_note(60)
    for _ in range(200):
        ga_output = np.concatenate((ga_output, poly.render()))
    
    ga_output = ga_output / np.max(np.abs(ga_output))
    return ga_output

def fitness(chromosome, ga_input):
    output = simulate(chromosome)
    S_output = librosa.feature.melspectrogram(y=output, sr=22050, n_mels=128, fmax=8000)
    S_input = librosa.feature.melspectrogram(y=ga_input, sr=22050, n_mels=128, fmax=8000)
    
    log_S_output = librosa.power_to_db(S_output, ref=np.max)
    log_S_input = librosa.power_to_db(S_input, ref=np.max)
    mel_mse = np.mean((log_S_output - log_S_input) ** 2)
    
    return mel_mse

def select(population, fitness_scores, num_parents):
    selected_indices = np.argsort(fitness_scores)[:num_parents]
    return [population[i] for i in selected_indices]

def genetic_algorithm(ga_input, population_size=10, generations=100, mutation_rate=0.1):
    population = initialize_population(population_size)
    for generation in range(generations):
        fitness_scores = [fitness(individual, ga_input) for individual in population]

        parents = select(population, fitness_scores, population_size // 2)
        
        offspring = []
        num_parents = len(parents)
        for _ in range(0, num_parents - num_parents % 2, 2):  # Ensure we only go up to the last even index
            child1, child2 = crossover(parents[_], parents[_+1])
            offspring.extend([child1, child2])

        # If the number of parents is odd, handle the last parent here
        if num_parents % 2 == 1:
            offspring.append(parents[-1])
            # alternatively
            # child1, child2 = crossover(parents[-1], parents[0])
            # offspring.extend([child1, child2])
        
        # Mutation
        for i in range(len(offspring)):
            if np.random.rand() < mutation_rate:
                offspring[i] = mutate(offspring[i])
        
        best_index = np.argmax(fitness_scores)
        best_individual = population[best_index]
        
        # Replacement
        worst_indices = np.argsort(fitness_scores)[-len(offspring):]
        for i, idx in enumerate(worst_indices):
            population[idx] = offspring[i]
        
        # Elitism
        if best_index in worst_indices:
            population[worst_indices[-1]] = best_individual
        else:
            population[np.argmin(fitness_scores)] = best_individual
        
        print(f"Generation {generation}: Best MSE = {min(fitness_scores)}")
    
    best_index = np.argmin([fitness(individual, ga_input) for individual in population])
    return population[best_index]

best = genetic_algorithm(ga_input, population_size=100, generations=15, mutation_rate=0.3)
print(best, decode_parameters(best))
ga_output = simulate(best)
scaled = np.int16(ga_output / np.max(np.abs(ga_output)) * 32767)
write('/tmp/ga_output.wav', 44100, scaled)