from pyNN.standardmodels import StandardIonChannelModel, build_translations
import pyNN.neuron as sim
from pyNN.morphology import NeuroMLMorphology, NeuriteDistribution, Morphology as Morph, IonChannelDistribution
from pyNN.parameters import IonicSpecies
from neuron import h, nrn, hclass
from pyNN.neuron.cells import RandomSpikeSource
from pyNN.standardmodels.cells import SpikeSourceGamma
from pyNN.neuron.simulator import state
import numpy as np



class NaChannel(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": 0.12, #uniform('all', 0.12),
        "e_rev": 50,
        "vt":-57.65,
    }
    
    default_initial_values = {
        'm': 1.0,  # initial value for gating variable m
        'h': 0.0,  # initial value for gating variable h
    }
    recordable = ['ina', 'm', 'h']
    units = {
        'ina': 'mA/cm2',
        'm': 'dimensionless',
        'h': 'dimensionless',
    }
    translations = build_translations(
        ('conductance_density', 'gna'),
        ('e_rev', 'ena'),
        ('vt', 'vt'),
    )
    variable_translations = {
        'h': ('motoneuron', 'h'),
        'm': ('motoneuron', 'm'),
        'ina': ('motoneuron', 'ina'),
    }
    model = "motoneuron"
    conductance_density_parameter = 'gna'
    def get_schema(self):
        return {
            "conductance_density": IonChannelDistribution,
            "e_rev": float,
            "vt": float
        } 
class PassiveLeak(StandardIonChannelModel):
        translations = build_translations(
            ('conductance_density', 'gl'),
            ('e_rev', 'el'),
        )
        variable_translations = {}
        model = "motoneuron"
        conductance_density_parameter = 'gl'

class cell_class(sim.MultiCompartmentNeuron):
        def __init__(self, **parameters):
    
            self.label = "mn1"
            self.ion_channels = {'pas_soma': PassiveLeak, 'pas_dend': sim.PassiveLeak,
                                 'na': NaChannel, 'kf': KfChannel, 'ks': KsChannel}
            self.units = {'v':'mV', 'gsyn_exc': 'uS', 'gsyn_inh': 'uS', 'na.m': 'dimensionless', 
                          'na.h': 'dimensionless', 'kf.n': 'dimensionless', 'ks.p': 'dimensionless', 
                          'na.ina': 'mA/cm2', 'kf.ikf': 'mA/cm2', 'ks.iks': 'mA/cm2'}
            self.post_synaptic_entities = {'syn': sim.CondExpPostSynapticResponse}
            
            super(cell_class, self).__init__(**parameters)

class KfChannel(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": 0.12, #uniform('all', 0.12),
        "e_rev": -80,
        "vt": -57.65
    }
    
    recordable = ['ikf','n'] 
    translations = build_translations(
        ('conductance_density', 'gk_fast'),
        ('e_rev', 'ekf'),
        ('vt', 'vt'),
    )
    variable_translations = {
        'n': ('motoneuron', 'n'),
    }
    default_initial_values = {
        'n': 1,  # initial value for gating variable m
    }
    units = {
        'ikf': 'mA/cm2',
        'n': 'dimensionless',
    }
    model = "motoneuron"
    conductance_density_parameter = 'gk_fast'
    def get_schema(self):
        return {
            "conductance_density": IonChannelDistribution,
            "e_rev": float,
            "vt": float
        }
class KsChannel(StandardIonChannelModel):
    default_parameters = {
        "conductance_density": 0.12, #uniform('all', 0.12),
        "e_rev": -80,
        "vt": -57.65
    }
    
    translations = build_translations(
        ('conductance_density', 'gk_slow'),
        ('e_rev', 'eks'),
        ('vt', 'vt'),
    )
    variable_translations = {
        'p': ('motoneuron', 'p'),
    }
    default_initial_values = {
        'p': 1,  # initial value for gating variable m
    }
    units = {
        'iks': 'mA/cm2',
        'p': 'dimensionless',
    }
    recordable = ['iks','p']
    model = "motoneuron"
    conductance_density_parameter = 'gk_slow'
    def get_schema(self):
        return {
            "conductance_density": IonChannelDistribution,
            "e_rev": float,
            "vt": float
        }
class SetRate(object):
    """
    A callback which changes the firing rate of a population of poisson
    processes at a fixed interval.
    """

    def __init__(self, population_source, population_neuron, interval=20.0):
        self.population_source = population_source
        self.population_neuron = population_neuron
        self.interval = interval
        
    def __call__(self, t):
        rate = (83+70* np.sin(25*t/1000))
        self.population_source.set(beta=rate)
        return t + self.interval
        
class RandomGammaStartSpikeSource(hclass(h.GammaProcess)):
    
    parameter_names = ('alpha', 'beta', 'start', 'duration')

    def __init__(self, alpha=1, beta=0.1, start=0, duration=0):
        self.alpha = alpha
        self.beta = beta
        self.start = start
        self.duration = duration
        self.spike_times = h.Vector(0)
        self.source = self
        self.rec = h.NetCon(self, None)
        self.switch = h.NetCon(None, self)
        self.source_section = None
        self.seed(state.mpi_rank + state.native_rng_baseseed)

    def __new__(cls, *arg, **kwargs):
        return super().__new__(cls, *arg, **kwargs)

class SpikeSourceGammaStart(SpikeSourceGamma):
    

    translations = build_translations(
        ('alpha',    'alpha'),
        ('beta',     'beta',    0.001),
        ('start',    'start'),
        ('duration', 'duration'),
    )
    model = RandomGammaStartSpikeSource

    # SetRate class to change the firing rate
class SetRate(object):
    def __init__(self, population_source, population_neuron, force_objects, interval=20.0):
        self.population_source = population_source
        self.population_neuron = population_neuron
        self.force_objects = force_objects  
        self.interval = interval

    def __call__(self, t):
        total_force = sum(force.F for force in self.force_objects.values())
        rate = 83 + ((200 - total_force) * 0.01)
        self.population_source.set(beta=rate)
    
        return t + self.interval
    
class SetRate(object):
    def __init__(self, population_source, population_neuron, force_objects, interval=20.0, ref=0):
        self.population_source = population_source
        self.population_neuron = population_neuron
        self.force_objects = force_objects  
        self.interval = interval
        self.ref = ref 
        print(f'valor: {self.ref}')

    def __call__(self, t):
        total_force = sum(force.F for force in self.force_objects.values())
        rate = 83 + ((self.ref - total_force) * 0.01)
        self.population_source.set(beta=rate)
        
        return t + self.interval
    
class SetRateIntControl(object):
    """
    A callback which changes the firing rate of a population of poisson
    processes at a fixed interval.
    """


    def __init__(self, population_source, population_neuron, force_objects, interval=20.0, ref=0):
        self.population_source = population_source
        self.population_neuron = population_neuron
        self.force_objects = force_objects  
        self.interval = interval
        self.ref = ref 
        self.Fint = 0
        self.F20 = 0
        self.F40 = 0
        self.F60 = 0
        self.F80 = 0
        print(f'valor: {self.ref}')

    def __call__(self, t):
        total_force = sum(force.F for force in self.force_objects.values())
        erro = (self.ref - self.F80)
        self.Fint =  self.Fint + erro*self.interval
        rate = 10+(erro*0.01 + 0.0003*self.Fint)
        self.population_source.set(beta=rate)
        self.F20 = total_force
        self.F40 = self.F20
        self.F60 = self.F40
        self.F80 = self.F60

        return t + self.interval

    








    

    













