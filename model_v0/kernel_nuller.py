import numpy as np
from utils import *

class KN0:
    def __init__(
            self,
            shifters_offset:list[float]=None,
            input_phase_rms:float=0.1,
            inner_phase_rms:float=0.1
        ):
        """--------------------------------------------------------------------
        Create a 4 telescope Kernel-Nuller (version 0)
        
        Parameters
        ----------
        - `shifters_offset` : array-like of 14 floats, the phase offsets of
        the phase shifters
        --------------------------------------------------------------------"""

        self.shifters_offset = shifters_offset
        self.input_phase_rms = input_phase_rms
        self.inner_phase_rms = inner_phase_rms

        self._shifters_noise = np.zeros(14)
        self.noise_all_shifters()

        self._shifters_noise = np.zeros(14)
        self._shifters_noise[0:4] = bound_phase(np.random.normal(scale=input_phase_rms, size=4))
        self._shifters_noise[4:] = bound_phase(np.random.normal(scale=inner_phase_rms, size=10))

        self._shifters_error = bound_phase(self._shifters_offset + self._shifters_noise)


    # Attribute getters and setters -------------------------------------------

    # Shifters offset
    @property
    def shifters_offset(self):
        return self._shifters_offset
    
    @shifters_offset.setter
    def shifters_offset(self, value:list[float]):
        self._shifters_offset = bound_phase(np.array(value, dtype=float))
        assert len(self.shifters_offset) == 14, "shift_powers must be a list of 14 floats"

    # Input phase RMS
    @property
    def input_phase_rms(self):
        return self._input_phase_rms
    
    @input_phase_rms.setter
    def input_phase_rms(self, value:float):
        self._input_phase_rms = float(value)

    # Inner phase RMS
    @property
    def inner_phase_rms(self):
        return self._inner_phase_rms
    
    @inner_phase_rms.setter
    def inner_phase_rms(self, value:float):
        self._inner_phase_rms = float(value)
    
    # Shifters noise
    @property
    def shifters_noise(self):
        return self._shifters_noise
    
    @shifters_noise.setter
    def shifters_noise(self, value:np.array):
        shifters_noise = bound_phase(np.array(value, dtype=float))
        assert len(shifters_noise) == 14, "shifters_noise must be a list of 14 floats"
        self._shifters_noise = shifters_noise

    def noise_all_shifters(self):
        self.noise_input_shifters()
        self.noise_inner_shifters()

    def noise_input_shifters(self):
        self._shifters_noise[0:4] = bound_phase(np.random.normal(scale=self.input_phase_rms, size=4))
    
    def noise_inner_shifters(self):
        self._shifters_noise[4:] = bound_phase(np.random.normal(scale=self.inner_phase_rms, size=10))
        
    # Main method -------------------------------------------------------------

    def __call__(self, beams:np.array, shifts:np.array=None, mode:str="numeric") -> tuple[complex, np.array, dict[np.array]]:
        """--------------------------------------------------------------------
        Simulate a 4 telescope Kernel-Nuller propagation
        
        Parameters
        ----------
        - `beams` : List of input beams complex amplitudes
        - `shifts` : List of input phase corrections
        - `mode` : String, the propagation mode. Can be either "numeric" or "analytic"
        
        Returns
        -------
        - Bright channel complex amplitude
        - List of dark channels complex amplitudes
        --------------------------------------------------------------------"""

        if mode.lower() == "numeric":
            return self.numeric(beams, shifts)
        elif mode.lower() == "analytic":
            return self.analytic(beams, shifts)
        else:
            raise ValueError("mode must be either 'numeric' or 'analytic'")

    def numeric(self, beams:np.array, shifts:np.array=None) -> tuple[complex, np.array, dict[np.array]]:
        """--------------------------------------------------------------------
        Simulate a 4 telescope Kernel-Nuller propagation using a numeric approach

        Parameters
        ----------
        - `beams` : List of input beams complex amplitudes
        - `shifts` : List of input phase corrections

        Returns
        -------
        - Bright channel complex amplitude
        - List of dark channels complex amplitudes
        --------------------------------------------------------------------"""

        # If no shift_powers are provided, use the default ones
        if shifts is None:
            shifts = np.zeros(14)
        
        # Add perturbations
        shifts = bound_phase(shifts + self.shifters_noise + self.shifters_offset)

        # First layer of pahse shifters
        nuller_inputs = phase_shift(beams, shifts[:4])

        # First layer of nulling
        N1_output = nuller_2x2(nuller_inputs[:2])
        N2_output = nuller_2x2(nuller_inputs[2:])

        # Second layer of phase shifters
        N1_output_shifted = phase_shift(N1_output, shifts[4:6])
        N2_output_shifted = phase_shift(N2_output, shifts[6:8])

        # Second layer of nulling
        N3_output = nuller_2x2(np.array([N1_output_shifted[0], N2_output_shifted[0]]))
        N4_output = nuller_2x2(np.array([N1_output_shifted[1], N2_output_shifted[1]]))

        bright_channel = N3_output[0]

        # Beam splitting
        S_inputs = np.array([
            N3_output[1],
            N3_output[1],
            N4_output[0],
            N4_output[0],
            N4_output[1],
            N4_output[1]
        ]) * 1/np.sqrt(2)

        # Last layer of phase shifters
        S_inputs = phase_shift(S_inputs, shifts[8:])

        # Beam mixing
        S1_output = splitmix_2x2(np.array([S_inputs[0],S_inputs[2]]))
        S2_output = splitmix_2x2(np.array([S_inputs[1],S_inputs[4]]))
        S3_output = splitmix_2x2(np.array([S_inputs[3],S_inputs[5]]))

        dark_channels = np.concatenate([S1_output, S2_output, S3_output])

        return bright_channel, dark_channels, {"inputs":beams, "first_nuller_layer":np.concatenate([N1_output,N2_output]),"second_nuller_layer":np.concatenate([N3_output,N4_output])}