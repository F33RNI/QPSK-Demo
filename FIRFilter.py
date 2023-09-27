"""
 Copyright (C) 2023 Fern Lane, QPSK-Demo project

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 See the License for the specific language governing permissions and
 limitations under the License.

 IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY CLAIM, DAMAGES OR
 OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np

FILTER_TYPE_LOWPASS = 0
FILTER_TYPE_HIGHPASS = 1
FILTER_TYPE_BANDPASS = 2


def estimate_filter_len(cutoff_frequency_low: float,
                        cutoff_frequency_high: float,
                        sample_rate: int,
                        attenuation: float):
    """
    Determines number of taps (filter length)
    :param cutoff_frequency_low: Low cutoff frequency (in Hz)
    :param cutoff_frequency_high: High cutoff frequency (in Hz)
    :param sample_rate: Sampling rate (in Hz)
    :param attenuation: Attenuation (in dB)
    :return: Filter size
    """
    return int(np.round(attenuation * sample_rate / (22 * (cutoff_frequency_high - cutoff_frequency_low))) - 1)


class FIRFilter:
    def __init__(self, pass_type: int,
                 sample_rate: int,
                 taps_n: int,
                 cutoff_frequency: float,
                 cutoff_frequency_2=None) -> None:
        """
        FIR Filter implementation
        Original code from: https://ryanclaire.blogspot.com/2020/09/fir-filter-c-python.html
        :param pass_type: FILTER_TYPE_LOWPASS, FILTER_TYPE_HIGHPASS or FILTER_TYPE_BANDPASS
        :param sample_rate: Sampling rate (in Hz)
        :param taps_n: Filter length (number of taps)
        :param cutoff_frequency: Low/High cutoff frequency (in Hz)
        :param cutoff_frequency_2: High cutoff frequency for bandpass filter (in Hz)
        """
        self._pass_type = pass_type
        self._sample_rate = sample_rate
        self._taps_n = taps_n
        self._cutoff_frequency = cutoff_frequency
        self._cutoff_frequency_2 = cutoff_frequency_2

        self.impulse_response = self.create_filter()
        self.filter_state = np.zeros(len(self.impulse_response), dtype=np.float32)

    def create_filter(self) -> np.ndarray:
        """
        Initializes filter
        :return: taps
        """
        taps = np.zeros(self._taps_n, dtype=np.float32)

        if self._pass_type == FILTER_TYPE_LOWPASS:
            fc = self._cutoff_frequency / self._sample_rate
            omega = 2 * np.pi * fc
        elif self._pass_type == FILTER_TYPE_HIGHPASS:
            fc = self._cutoff_frequency / self._sample_rate
            omega = 2 * np.pi * fc
        else:
            flc = self._cutoff_frequency / self._sample_rate
            fhc = self._cutoff_frequency_2 / self._sample_rate
            omega_l = 2 * np.pi * flc
            omega_h = 2 * np.pi * fhc

        middle = int(self._taps_n / 2)
        for i in range(self._taps_n):
            if self._pass_type == FILTER_TYPE_LOWPASS:
                if i == middle:
                    taps[i] = 2 * fc
                else:
                    taps[i] = np.sin(omega * (i - middle)) / (np.pi * (i - middle))
            elif self._pass_type == FILTER_TYPE_HIGHPASS:
                if i == middle:
                    taps[i] = 1 - 2 * fc
                else:
                    taps[i] = - np.sin(omega * (i - middle)) / (np.pi * (i - middle))
            else:
                if i == middle:
                    taps[i] = 2 * fhc - 2 * flc
                else:
                    taps[i] = np.sin(omega_h * (i - middle)) / (np.pi * (i - middle)) \
                              - np.sin(omega_l * (i - middle)) / (np.pi * (i - middle))

        return taps

    def filter(self, input_value: np.float32):
        """
        Filters input_value
        :param input_value: value to filter
        :return: filtered value
        """
        self.filter_state[:-1] = self.filter_state[1:]
        self.filter_state[-1] = input_value

        # Convolve the impulse response with the shifted state
        return np.sum(self.filter_state * self.impulse_response)
