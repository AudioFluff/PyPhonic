import os
import random
from datetime import datetime

import numpy as np
from scipy.io.wavfile import write
import librosa

from pyphonic.preset_19_synth import Poly, Synth, get_preset, reset_wavetables, get_wavetables

MAX_PARAMS = 13

# poly = Poly(**get_preset("cello"))
poly = Poly(stack=[Synth(op="randomwalk", detune_coarse=0, detune=1, rel_vel=0.5, phase=20, attack=0.1, decay=0.1, sustain=0.5, release=30)])
# poly = Poly(stack=[Synth(op="sin", detune_coarse=0, detune=1, rel_vel=0.5, phase=20, attack=0.1, decay=0.1, sustain=0.5, release=30),
#                    Synth(op="tri", detune_coarse=0, detune=0, rel_vel=0.5, phase=0),
#                    Synth(op="noise", detune_coarse=0, detune=0, rel_vel=0.1, phase=0),
#                    ])
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
        (osc_type, detune_coarse, detune, rel_vel, phase,
         attack, decay, sustain, release,
         delay_seconds, delay_length, delay_feedback, delay_mix
        ) = oscillators[i]
        params.extend([osc_type, detune_coarse, detune, rel_vel, phase,
                       attack, decay, sustain, release,
                       delay_seconds, delay_length, delay_feedback, delay_mix])
    return params

def decode_parameters(chromosome):
    num_oscillators = chromosome[0]
    ops = {
        0: "square",
        1: "saw",
        2: "tri",
        3: "sin",
        4: "noise",
        5: "laser",
        6: "randomwalk"
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
        synth["delay_seconds"] = chromosome[i + 9]
        synth["delay_length"] = chromosome[i + 10]
        synth["delay_feedback"] = chromosome[i + 11]
        synth["delay_mix"] = chromosome[i + 12]
        stack.append((Synth, synth))
    return stack

def initialize_population(size):
    population = []
    for _ in range(size):
        num_oscillators = np.random.randint(1, 5)  # 1-4 oscillators
        oscillators = []
        for _ in range(num_oscillators):
            osc_type = np.random.randint(0, 7)
            detune_coarse = int(np.random.triangular(-12, 0, 12))
            detune_fine = int(np.random.triangular(-100, 0, 101))
            rel_vel = np.round(np.random.uniform(0.1, 1.0), 2)
            phase = int(np.random.triangular(-360, 0, 361))
            attack = np.round(np.random.triangular(0, 0, 1), 2)
            decay = np.round(np.random.triangular(0, 0, 1), 2)
            sustain = np.round(np.random.triangular(0, 0.5, 1), 2)
            release = np.round(np.random.triangular(0, 0, 4), 2)
            delay_seconds = np.round(np.random.triangular(0, 0, 2), 2)
            delay_length = np.round(np.random.triangular(0, 0, 5), 2)
            delay_feedback = np.round(np.random.triangular(0, 0, 1), 2)
            delay_mix = np.round(np.random.triangular(0, 0, 1), 2)
            oscillators.append([osc_type, detune_coarse, detune_fine, rel_vel, phase,
                                attack, decay, sustain, release,
                                delay_seconds, delay_length, delay_feedback, delay_mix])
        chromosome = encode_parameters(num_oscillators, oscillators)
        assert len(chromosome) == 1 + (num_oscillators * MAX_PARAMS)
        population.append((chromosome, {}))
    return population

def mutate(chromosome):
    chromosome, extra = chromosome
    mutation_point = np.random.randint(0, len(chromosome))
    if mutation_point == 0:  # Number of oscillators
        values = np.array([1, 2, 3, 4])
        weights = np.array([0.5, 0.3, 0.15, 0.05])
        weights /= weights.sum()
        value = np.random.choice(values, p=weights)
        if value > chromosome[mutation_point]: ## Need to add params for another osc
            for _ in range(value - chromosome[mutation_point]):
                osc_type = np.random.randint(0, 7)
                detune_coarse = int(np.random.triangular(-12, 0, 12))
                detune_fine = int(np.random.triangular(-100, 0, 101))
                rel_vel = np.round(np.random.uniform(0.1, 1.0), 2)
                phase = int(np.random.triangular(-360, 0, 361))
                attack = np.round(np.random.triangular(0, 0, 1), 2)
                decay = np.round(np.random.triangular(0, 0, 1), 2)
                sustain = np.round(np.random.triangular(0, 0.5, 1), 2)
                release = np.round(np.random.triangular(0, 0, 4), 2)
                delay_seconds = np.round(np.random.triangular(0, 0, 2), 2)
                delay_length = np.round(np.random.triangular(0, 0, 5), 2)
                delay_feedback = np.round(np.random.triangular(0, 0, 1), 2)
                delay_mix = np.round(np.random.triangular(0, 0, 1), 2)
                chromosome.extend([osc_type, detune_coarse, detune_fine, rel_vel, phase,
                                   attack, decay, sustain, release,
                                   delay_seconds, delay_length, delay_feedback, delay_mix])
        elif value < chromosome[mutation_point]: ## Need to remove params for another osc
            for _ in range(chromosome[mutation_point] - value):
                chromosome = chromosome[:1 + (value * MAX_PARAMS)]
        chromosome[mutation_point] = value
        chromosome = chromosome[:1 + (value * MAX_PARAMS)]
        
    else:  # Oscillator parameters
        
        parameter = (mutation_point - 1) % MAX_PARAMS

        if parameter == 0:  # Oscillator type
            chromosome[mutation_point] = np.random.randint(0, 7)
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
        elif parameter == 9:  # Delay Seconds
            chromosome[mutation_point] = np.round(np.random.triangular(0, 0, 2), 2)
        elif parameter == 10:  # Delay Length
            chromosome[mutation_point] = np.round(np.random.triangular(0, 0, 5), 2)
        elif parameter == 11:  # Delay Feedback
            chromosome[mutation_point] = np.round(np.random.triangular(0, 0, 1), 2)
        elif parameter == 12:  # Delay Mix
            chromosome[mutation_point] = np.round(np.random.triangular(0, 0, 1), 2)
    
    assert len(chromosome) == 1 + (chromosome[0] * MAX_PARAMS)

    if extra.get("randomwalk_waveforms") is not None:
        for i in range(1, len(chromosome), MAX_PARAMS):
            if chromosome[i] == 6 and extra["randomwalk_waveforms"].get(i) is not None:
                if random.choice([1,2]) == 1 or np.sum(np.abs(extra["randomwalk_waveforms"][i])) < 0.1:
                    del extra["randomwalk_waveforms"][i]
                else:
                    extra["randomwalk_waveforms"][i] += np.random.normal(size=extra["randomwalk_waveforms"][i].size)
                ## TODO muttate the waveform

    return (chromosome, extra)

def crossover(parent1, parent2):
    parent1, extra1 = parent1
    parent2, extra2 = parent2
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

    for i in range(1, len(parent1), MAX_PARAMS):
        if child1[i] == 6 and child2[i] == 6 and parent1[i] == 6 and parent2[i] == 6:
            print("Both parents, and both children, are randomwalks")
            option = random.choice([1,2,3,4])
            if option == 1:
                # pass it down directly
                extra1["randomwalk_waveforms"] = extra1.get("randomwalk_waveforms", {})
                extra2["randomwalk_waveforms"] = extra2.get("randomwalk_waveforms", {})
            elif option == 2:
                # swap them
                extra1["randomwalk_waveforms"] = extra2.get("randomwalk_waveforms", {})
                extra2["randomwalk_waveforms"] = extra1.get("randomwalk_waveforms", {})
            elif option == 3:
                # drop them
                extra1["randomwalk_waveforms"] = {}
                extra2["randomwalk_waveforms"] = {}
            elif option == 4:
                # add both parents' waveforms together and normalize. If one is shorter, add from the other
                p1_waveforms = extra1.get("randomwalk_waveforms", {})
                p2_waveforms = extra2.get("randomwalk_waveforms", {})
                if i in p1_waveforms and i in p2_waveforms:
                    if p1_waveforms[i].size == p2_waveforms[i].size:
                        extra1["randomwalk_waveforms"] = {i: (p1_waveforms[i] + p2_waveforms[i]) / 2}
                        extra2["randomwalk_waveforms"] = {i: (p1_waveforms[i] + p2_waveforms[i]) / 2}
                    else:
                        if p1_waveforms[i].size > p2_waveforms[i].size:
                            p2_waveforms[i] = librosa.resample(p2_waveforms[i], orig_sr=44100, target_sr=44100 * p1_waveforms[i].size / p2_waveforms[i].size)
                        else:
                            p1_waveforms[i] = librosa.resample(p1_waveforms[i], orig_sr=44100, target_sr=44100 * p2_waveforms[i].size / p1_waveforms[i].size)
                        if p1_waveforms[i].size != p2_waveforms[i].size:
                            if p1_waveforms[i].size > p2_waveforms[i].size:
                                p1_waveforms[i] = p1_waveforms[i][:p2_waveforms[i].size]
                            else:
                                p2_waveforms[i] = p2_waveforms[i][:p1_waveforms[i].size]
                        extra1["randomwalk_waveforms"] = {i: (p1_waveforms[i] + p2_waveforms[i]) / 2}
                        extra2["randomwalk_waveforms"] = {i: (p1_waveforms[i] + p2_waveforms[i]) / 2}
    return (child1, extra1), (child2, extra2)


def simulate(chromosome):
    chromosome, extra = chromosome
    stack = []

    reset_wavetables()
    assert not get_wavetables()

    for i, item in enumerate(decode_parameters(chromosome)):
        synth, params = item
        if params["op"] == "randomwalk":
            if extra.get("randomwalk_waveforms") is not None:
                if extra["randomwalk_waveforms"].get(i) is not None:
                    def noteToFreq(midi_note):
                        a = 440
                        freq = (a / 32) * (2 ** ((midi_note - 9) / 12))
                        return round(freq, 2)
                    orig_hz = noteToFreq(60 + params["detune_coarse"]) + params["detune"]
                    params["op_extra_params"] = {"wavetable": {orig_hz: extra["randomwalk_waveforms"].get(i)}}
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
    poly.stop_note(60)

    for i, item in enumerate(stack):
        if stack[i].op == "randomwalk":
            extra["randomwalk_waveforms"] = extra.get("randomwalk_waveforms", {})
            extra["randomwalk_waveforms"][i] = poly.synths[60][i].waveform
        new_chromosome = chromosome, extra

    poly.start_note(60, 10)

    ga_output = np.array([], dtype=np.float32)

    for _ in range(200):
        ga_output = np.concatenate((ga_output, poly.render()), dtype=np.float32)

    poly.stop_note(60)
    for _ in range(200):
        ga_output = np.concatenate((ga_output, poly.render()), dtype=np.float32)
    
    if np.max(np.abs(ga_output)) > 0:
        ga_output = ga_output / np.max(np.abs(ga_output))
    
    return ga_output, new_chromosome

def fitness(chromosome, ga_input):
    output, chromosome = simulate(chromosome)
    try:
        S_output = librosa.feature.melspectrogram(y=output, sr=22050, n_mels=128, fmax=8000)
        S_input = librosa.feature.melspectrogram(y=ga_input, sr=22050, n_mels=128, fmax=8000)
    except Exception as e:
        print(f"Error rendering audio {str(e)}")
        print(decode_parameters(chromosome))
        return 1e5
    
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

start_time = datetime.now()
best = genetic_algorithm(ga_input, population_size=15, generations=30, mutation_rate=0.4)
print(f"Time taken (seconds): {(datetime.now() - start_time).total_seconds()}")
best_synth, best_extra = best
print(best_synth, decode_parameters(best_synth))
ga_output, unnecessary_update_chromosome = simulate(best)
if np.max(np.abs(ga_output)) > 0:
    scaled = np.int16(ga_output / np.max(np.abs(ga_output)) * 32767)
    write('/tmp/ga_output.wav', 44100, scaled)
else:
    print("Output was all zeros.")