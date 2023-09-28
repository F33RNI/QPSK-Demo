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


class RRCFilter:
    def __init__(self, pass_type: int,
                 sample_rate: int,
                 cutoff_frequency: float,
                 cutoff_frequency_2: float,
                 positive_lobes_n: int,
                 alpha=0.5) -> None:
        """
        Root raised cosine (RRC) low-pass filter implementation by Fern Lane
        :param pass_type: FILTER_TYPE_LOWPASS, FILTER_TYPE_HIGHPASS or FILTER_TYPE_BANDPASS
        :param sample_rate: Sampling rate (in Hz)
        :param cutoff_frequency: Low/High cutoff frequency (in Hz)
        :param cutoff_frequency_2: High cutoff frequency for bandpass filter (in Hz) (set to None in band-pass filter)
        :param positive_lobes_n: Used for calculating filter length (use something between 5 and 50)
        :param alpha: Roll-off factor (from 0 to 0.999 (steepest))
        """
        self._pass_type = pass_type
        self._sample_rate = sample_rate
        self._cutoff_frequency = cutoff_frequency
        self._cutoff_frequency_2 = cutoff_frequency_2
        self._positive_lobes_n = positive_lobes_n
        self._alpha = alpha

        # Calculate band-pass parameters
        center_frequency = None
        bandwidth = None
        if self._pass_type == FILTER_TYPE_BANDPASS:
            center_frequency = (self._cutoff_frequency + self._cutoff_frequency_2) / 2
            bandwidth = self._cutoff_frequency_2 - self._cutoff_frequency

        # Calculate filter length
        # TODO: make better for bandpass
        if self._pass_type == FILTER_TYPE_BANDPASS:
            self.filter_length = int(self._sample_rate /
                                     (2 * ((self._cutoff_frequency + self._cutoff_frequency_2) / 2))) \
                                 * (self._positive_lobes_n * 2 + 1)
        else:
            self.filter_length = int(self._sample_rate / (2 * self._cutoff_frequency)) \
                                 * (self._positive_lobes_n * 2 + 1)

        # Generate impulse response
        if self._pass_type == FILTER_TYPE_LOWPASS:
            self.impulse_response = self.create_filter(self._cutoff_frequency)
        elif self._pass_type == FILTER_TYPE_HIGHPASS:
            self.impulse_response = self.create_filter(self._cutoff_frequency)
            delta_function = np.zeros(self.filter_length, dtype=np.float32)
            delta_function[delta_function.shape[0] // 2] = 1.
            self.impulse_response = delta_function - self.impulse_response
        else:
            impulse_response_lower = self.create_filter(center_frequency - bandwidth / 2)
            impulse_response_upper = self.create_filter(center_frequency + bandwidth / 2)
            self.impulse_response = impulse_response_lower - impulse_response_upper

        # Initialize rotating buffer for future convolution
        self.filter_state = np.zeros(self.impulse_response.shape[0], dtype=np.float32)

    def create_filter(self, cutoff_frequency) -> np.ndarray:
        """
        Initializes filter's low-pass impulse response
        Formula from: https://en.wikipedia.org/wiki/Root-raised-cosine_filter
        :return:
        """
        # Allocate impulse_response
        impulse_response = np.zeros(self.filter_length, dtype=np.float32)

        # Calculate Ts
        # NOTE: cutoff_frequency * self._alpha * np.sqrt(self._alpha)
        # is a compensation for alpha (selected experimentally)
        ts = 1 / (cutoff_frequency + cutoff_frequency * self._alpha * np.sqrt(self._alpha))

        # Invert alpha to calculate beta
        beta = 1. - self._alpha

        # Calculate impulse response
        for i in range(0, len(impulse_response)):
            # Make "virtual" 0 at the center
            t = (-len(impulse_response) / 2 + i) / self._sample_rate

            # t = 0
            if t == 0:
                term_0 = 1 / ts
                term_1 = (1 + beta * (4 / np.pi - 1))
                impulse_response[i] = term_0 * term_1 / self._sample_rate
            # t = +/- ts/(4 * beta)
            elif abs(t) == (ts / (4 * beta)):
                term_0 = (beta / (ts * np.sqrt(2)))
                term_1 = (1 + 2 / np.pi)
                term_2 = np.sin(np.pi / (4 * beta))
                term_3 = (1 - 2 / np.pi)
                term_4 = np.cos(np.pi / (4 * beta))
                impulse_response[i] = term_0 * (term_1 * term_2 + term_3 * term_4) / self._sample_rate
            # otherwise
            else:
                term_0 = 1 / ts
                term_1 = np.sin(np.pi * (t / ts) * (1 - beta))
                term_2 = 4 * beta * (t / ts) * np.cos(np.pi * (t / ts) * (1 + beta))
                term_3 = np.pi * (t / ts)
                term_4 = 1 - np.power(4 * beta * (t / ts), 2)
                impulse_response[i] = term_0 * (term_1 + term_2) / (term_3 * term_4) / self._sample_rate

        return impulse_response

    def filter(self, input_value: np.float32):
        """
        Filters input_value (continues convolution)
        :param input_value: value to filter
        :return: filtered value
        """
        self.filter_state[:-1] = self.filter_state[1:]
        self.filter_state[-1] = input_value

        # Convolve the impulse response with the shifted state
        return np.sum(self.filter_state * self.impulse_response)
