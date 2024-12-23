import numpy as np
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt

# https://towardsdatascience.com/fourier-transform-the-practical-python-implementation-acdd32f1b96a

class Signal:


    def __init__(self, amplitude=1, frequency=10, duration=1, sampling_rate=1000, phase=0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.duration = duration
        self.phase = phase
        self.sampling_rate = sampling_rate
        self.time = np.linspace(0, self.duration, self.duration * self.sampling_rate, endpoint=False)
    
    def sine_wave(self):
        """
        y(t) = A sin(2pi x f x t  + phase)
        """
        return self.amplitude * np.sin(2 * np.pi * self.frequency * self.time   + self.phase)


    def cosine_wave(self):
        return self.amplitude * np.cos(2 * np.pi * self.frequency * self.time + self.phase)
    
    def square_wave(self):
        return self.amplitude * signal.square(2 * np.pi * self.frequency * self.time)

    def _exp(self):
        return np.exp(-2 * self.time)
    
    def damping_signal(self, signal):
        return self._exp() * signal
    
    def noisy_sigal(self, location, scale, signal):
        noise = np.random.normal(location, scale, size=signal.shape)
        return signal + noise

    def pulsated_sinsoidal(self, pulse_signal, sinusoidal):
        return pulse_signal + sinusoidal



frequeny = 5 
sampling_rate = 1000
duration = 2
amplitude = 2

sg = Signal(5,amplitude, duration, sampling_rate)
sinewave = sg.sine_wave()
dampped_sine = sg.damping_signal(sinewave)
noise = sg.noisy_sigal(2,1, dampped_sine)
time = np.linspace(0, duration, duration * sampling_rate)

#square = sg.square_wave()
#pulsated = sg.pulsated_sinsoidal(square, sinewave)

#plt.plot(time,noise)
#plt.show()


fourier = fft(noise)
plt.plot(np.abs(fourier))
plt.show()