from neuroml import Morphology, Segment, Point3DWithDiam as P
from pyNN.morphology import NeuroMLMorphology, NeuriteDistribution, Morphology as Morph, IonChannelDistribution
import pyNN.neuron as sim
import numpy as np

def create_somas(n):
    somas = []
    for i in range(n):
        diameter = 77.5 + i * ((82.5-77.5)/n)
        y = 18- i * (18/n)
        soma= Segment(proximal=P(x=diameter, y=y, z=0, diameter=diameter),
                       distal=P(x=0, y=0, z=0, diameter=diameter),
                       name="soma", id=0)
        somas.append(soma)
    return somas

def create_dends(n,somas):
     dends=[]
     for i in range(n):
        y = 18- i * (18/n)
        diameter = 41.5 + i * ((62.5-41.5)/n)
        x_distal= -5500 + i * ((-6789+5500)/n)
        dend = Segment(proximal=P(x=0, y=y, z=0, diameter=diameter),
               distal=P(x=x_distal, y=y, z=0, diameter=diameter),
               name="dendrite",
               parent=somas[i], id=1)
        dends.append(dend)
     return dends

def soma_dend(somas, dends):
    combined= []
    first_soma = somas[0]
    for i in range(len(dends)):
        combineds=NeuroMLMorphology(Morphology(segments=(somas[i], 
                                                dends[i])))
        combined.append(combineds)
    return combined

def generate_spike_times(i):
    input_rate = 83
    Tf = 100
    number = int(Tf * input_rate / 1000.0)
    gen = lambda: sim.Sequence(np.add.accumulate(np.random.exponential(1000.0 / input_rate, size=number)))
    if hasattr(i, "__len__"):
        return [gen() for j in i]
    else:
        return gen()

# somas= create_somas(n)
# dendrites = create_dends(n,somas)
# neurons= soma_dend(somas, dendrites) 

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def plot_disparos_neuronios(spiketrains, neuronio, delta_t=0.00005, filtro_ordem=4, freq_corte=0.001, tempo_max=1000):
    """
    Function that generates the Dirac impulse for the firing times of a neuron.
        Parameters:
        spiketrains: List of neuron firing trains.
        neuronio: Index of the neuron to be processed.
        delta_t: Time interval. 
        filtro_ordem : Order of the Butterworth filter. 
        freq_corte: Normalized cut-off frequency for the Butterworth filter.
        tempo_max: Maximum time for the x-axis (in milliseconds). 
    """
    
    # Array com os tempos de disparo do neurônio
    tempos_neuronios = spiketrains

    # Criação do vetor de tempo
    t = np.arange(0, tempo_max, delta_t)
    impulso_dirac = np.zeros_like(t)

    # Adiciona o impulso de Dirac em cada tempo de disparo do neurônio
    for tempo in tempos_neuronios:
        idx = np.argmin(np.abs(t - tempo/1000))  # encontra o índice mais próximo do tempo de disparo
        impulso_dirac[idx] = 1 / delta_t

    # Filtro Butterworth
    b, a = signal.butter(filtro_ordem, freq_corte)

    # Aplicação do filtro
    filtered_impulso = signal.filtfilt(b, a, impulso_dirac)

    # Plotar os resultados
    # plt.plot(t, filtered_impulso, label="Disparo do Neurônio (Filtrado)")
    # plt.title("Tempos de Disparo do Neurônio (Filtrado)")
    # plt.xlabel("Tempo (ms)")
    # plt.ylabel("Amplitude")
    # plt.xlim(0, tempo_max)
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(t, impulso_dirac, label="Disparo do Neurônio (Impulso de Dirac)")
    # plt.title("Tempos de Disparo do Neurônio (Impulso de Dirac)")
    # plt.xlabel("Tempo (ms)")
    # plt.ylabel("Amplitude")
    # plt.xlim(0, tempo_max)
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    return filtered_impulso, t
#plot_disparos_neuronios(data.spiketrains, neuronio=1)

# def neuromuscular_system(cells, n):   
#     muscle_units = dict()
#     force_objects = dict()
#     neuromuscular_junctions = dict()

#     for i in range(n):
#         muscle_units[i] = h.Section(name=f'mu{i}')
#         if calcium:
#             force_objects[i] = h.muscle_unit_calcium(muscle_units[i](0.5))
#         else:
#             force_objects[i] = h.muscle_unit(muscle_units[i](0.5))
#         neuromuscular_junctions[i] = h.NetCon(cells.all_cells[i]._cell.sections[0](0.5)._ref_v, force_objects[i], sec=cells.all_cells[i]._cell.sections[0])
        
#         force_objects[i].Fmax = 0.03 + (3 - 0.03)*i/n
#         force_objects[i].Tc = 140 + (96 - 140)*i/n
    
    # return muscle_units, force_objects, neuromuscular_junctions

def neuromuscular_system(cells, n, h, Umax = 1000):   
    muscle_units = dict()
    force_objects = dict()
    neuromuscular_junctions = dict()

    for i in range(n):
        muscle_units[i] = h.Section(name=f'mu{i}')
        force_objects[i] = h.muscle_unit_calcium(muscle_units[i](0.5))
        neuromuscular_junctions[i] = h.NetCon(cells.all_cells[i]._cell.sections[0](0.5)._ref_v, force_objects[i], sec=cells.all_cells[i]._cell.sections[0])
        
        force_objects[i].Fmax = 0.03 + (3 - 0.03) * i / n
        force_objects[i].Tc = 140 + (96 - 140) * i / n
        force_objects[i].Umax =  Umax
        neuromuscular_junctions[i].delay = 0.86/(44+9/n*i)*1000
        
    
    return muscle_units, force_objects, neuromuscular_junctions

def soma_força(force_objects, h, f):
    max_len = max(len(f[i]) for i in force_objects.keys())
    
   #Creates a vector to store the total force over time
    forca_total = h.Vector(max_len)
    forca_total.fill(0)  # Initializes with zeros
    
  # Sum the forces of all motor units
    for i in force_objects.keys():
        forca_individual = f[i]
        forca_total.add(forca_individual)  # Add each force vector to the total
        
    return forca_total




