import numpy as np
from scipy.signal import butter, lfilter
from collections import deque

class LowPassFilter:
    def __init__(self, cutoff, fs, order=4, window_size=30):
        """
        Initializes the low-pass filter for real-time use with a sliding window for each state.
        
        Parameters:
        - cutoff (float): The cutoff frequency of the filter in Hz.
        - fs (float): The sampling frequency in Hz.
        - order (int): The order of the Butterworth filter (default is 4).
        - window_size (int): The number of recent values to keep for the filter (default is 30).
        """
        ## Normalize the cutoff frequency to Nyquist frequency (fs/2)
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        
        ## Get the filter coefficients
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)
        
        ## Initialize a buffer (deque) for each state, with a fixed window size
        self.window_size = window_size
        self.input_buffers = [deque(maxlen=self.window_size) for _ in range(4)]  # Buffer for each of the 4 states
        self.previous_outputs = [None] * 4  # Stores previous outputs for each state

    def update(self, states):
        """
        Updates the filter with new states and returns the filtered states.
        
        Parameters:
        - states (list): A list of 4 input states.
        
        Returns:
        - cleaned_states (list): A list of the 4 filtered output states.
        """
        cleaned_states = []
        
        ## Loop over each of the 4 states
        for i in range(4):
            ## Add the new state to the respective buffer
            self.input_buffers[i].append(states[i])

            ## If we have enough data in the buffer, apply the filter
            if len(self.input_buffers[i]) >= self.window_size:
                ## Apply the filter using lfilter
                if self.previous_outputs[i] is None:
                    zi = [0.0] * (max(len(self.b), len(self.a)) - 1)  ## Initializing zi if it is None
                else:
                    zi = self.previous_outputs[i]
                
                y, self.previous_outputs[i] = lfilter(self.b, self.a, list(self.input_buffers[i]), zi=zi)
                
                ## Append the latest filtered value for the state
                cleaned_states.append(y[-1])  # Get the latest value
            else:
                ## If not enough data yet, return the unfiltered state
                cleaned_states.append(states[i])
        
        return cleaned_states
