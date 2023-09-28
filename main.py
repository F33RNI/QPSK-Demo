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

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import ButterworthFilter
import RRCFiler


# ################## #
# TRANSMITTER CONFIG #
# ################## #
# Data to encode and modulate
def string_to_qpsk_array(input_string):
    """
    Converts string to list of numbers 0-3 (pair of bits)
    :param input_string: Hello! (01001000 01100101 01101100 01101100 01101111 00100001)
    :return: [1, 0, 2, 0, 1, 2, 1, 1, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 3, 0, 2, 0, 1]
    """
    qpsk_array = []
    for char in input_string:
        binary_rep = format(ord(char), "08b")  # Convert character to 8-bit binary
        for i in range(0, len(binary_rep), 2):
            # Map pairs of bits to QPSK values (00 -> 0, 01 -> 1, 10 -> 2, 11 -> 3)
            pair = binary_rep[i:i + 2]
            if pair == "00":
                qpsk_array.append(0)
            elif pair == "01":
                qpsk_array.append(1)
            elif pair == "10":
                qpsk_array.append(2)
            elif pair == "11":
                qpsk_array.append(3)
    return qpsk_array


# Possible values:
# <0 - OFF (silence)
# 0 - 00
# 1 - 01
# 2 - 10
# 3 - 11
#
# [-1] - Silence,
# [0] * 60 - PLL sync burst,
# [0, 1, 2, 3] - phase (IQ) sync (MUST BE EXACTLY THESE VALUES IN THIS ORDER),
# string_to_qpsk_array("Hello!") - Data (payload),
# [-1] * 20 - Silence
#
# For PLL calibration use TX_DATA = [0] * 60
TX_DATA = [-1] + [0] * 60 + [0, 1, 2, 3] + string_to_qpsk_array("Hello!") + [-1] * 20

# Transmitter (modulator) carrier frequency (in Hz)
TRANSMITTER_CARRIER_FREQUENCY = 1000

# Amplitude of signal
CARRIER_AMPLITUDE = 0.8

# ############### #
# RECEIVER CONFIG #
# ############### #
# Receiver (demodulator) carrier frequency (in Hz)
# Make it different from TRANSMITTER_CARRIER_FREQUENCY to test PLL
RECEIVER_CARRIER_FREQUENCY = 950

# ################# #
# DATA RATES CONFIG #
# ################# #
# Sampling range (in Hz)
SAMPLING_RATE = 8000

# Rate of symbols in carrier halfcycles (2 halfcycles = 1 full carrier wave cycle)
# Minimum value: 1
# Lower values - each symbol (2 bits) takes less time (lower stability, higher data rate)
# Higher values - each symbol (2 bits) takes more time (higher stability, lower data rates)
HALFCYCLES_PER_SYMBOL = 4

# Signal bandwidth
# Half-cycles rate = TRANSMITTER_CARRIER_FREQUENCY * 2
# Symbol rate = TRANSMITTER_CARRIER_FREQUENCY * 2 / HALFCYCLES_PER_SYMBOL
# Bitrate = TRANSMITTER_CARRIER_FREQUENCY * 2 / HALFCYCLES_PER_SYMBOL * 2
BANDWIDTH = TRANSMITTER_CARRIER_FREQUENCY * 2 / HALFCYCLES_PER_SYMBOL * 2

# ############## #
# FILTERS CONFIG #
# ############## #
# Transmitter RRC output filter number of positive lobes and alpha (roll-off factor)
TRANSMITTER_FILTER_LOBES_N = 40
TRANSMITTER_FILTER_ALPHA = 0.98

# Transmitter RRC output filter number of positive lobes and alpha (roll-off factor)
RECEIVER_FILTER_LOBES_N = 40
RECEIVER_FILTER_ALPHA = 0.98

# ######################## #
# TRANSMISSION LINE CONFIG #
# ######################## #
# Set to True - to simulate transmission instead
# Set to False - to use your real speakers and mic
USE_LINE_SIMULATION = True

# Size of buffer for speakers and mic (PyAudio)
# Possible values: 64 / 128 / 256 / 512 / 1024 / etc...
CHUNK_SIZE = 256

# Maximum number of chunks to playback / record to prevent freezing if no data was detected
PLAYBACK_MAX_CHUNKS = 100

# Level of added noise (0 - 1)
NOISE_LEVEL = 0.05

# Low pass filter of line simulator (aka real bandwidth of signal) (in Hz).
LINE_LFP_FREQUENCY = 4000

# Add initial delay (in s)
LINE_DELAY_FROM = 0
LINE_DELAY_TO = 0.01

# ################################ #
# DIGITAL PHASE-LOCKED LOOP CONFIG #
# ################################ #
# Proportional gain term of PLL's PI controller
PLL_K_P = 0.2

# Integral gain term of PLL's PI controller
PLL_K_I = 8

# Gain of PLL's VCO (voltage to frequency transfer coefficient)
# Usually slightly less than half of carrier frequency
PLL_K_VCO = (RECEIVER_CARRIER_FREQUENCY / 2)

# ############################# #
# AUTOMATIC GAIN CONTROL CONFIG #
# ############################# #
# AGC accumulator K when signal <= accumulator
AGC_CONSTANT_FALLING = 0.98

# AGC accumulator K when signal > accumulator
AGC_CONSTANT_RISING = 0.98

# "Target" amplitude after AGC
AGC_CORRECTION_FACTOR = 0.7

# Signal (AGS accumulator) must be above this threshold to start PLL locking and demodulating (in dBFS)
CARRIER_START_AMPLITUDE = -20

# Signal (AGS accumulator) must be below this threshold to stop PLL locking and demodulating (in dBFS)
CARRIER_LOST_AMPLITUDE = -30

# ############### #
# PLL LOCK CONFIG #
# ############### #
# PLL's input (IQ error) should't change during PLL_LOCKED_TIME more than this value
PLL_LOCKED_THRESHOLD = 0.08

# PLL's input (IQ error) should't change during this time (in seconds) more than PLL_LOCKED_THRESHOLD
# Usually 20 full carrier cycles if enough
# Make sure to add [0] * 60 in TX_DATA (3 x 20)
PLL_LOCKED_TIME = 20 * (1 / RECEIVER_CARRIER_FREQUENCY)


class DPLL:
    def __init__(self, k_p: float, k_i: float, k_vco: float, frequency_filter_k=0.994) -> None:
        self._k_p = k_p
        self._k_i = k_i
        self._k_vco = k_vco
        self._frequency_filter_k = frequency_filter_k

        self.integrator = 0.
        self.output_phase = 0.
        self.vco = np.exp(1j * 0)
        self.frequency = RECEIVER_CARRIER_FREQUENCY
        self.frequency_filtered = RECEIVER_CARRIER_FREQUENCY
        self.zero_crossing_flag = False
        self.omega = 0
        self._omega_mod_prev = 0

    def step(self, error, sample_time) -> None:
        """
        Calculates one step of PLL
        :param error: input error
        :param sample_time: current sample time (in s)
        :return:
        """
        # Calculate integral controller
        self.integrator += self._k_i * error * (1 / SAMPLING_RATE)

        # Calculate PI output using proportional controller and integral controller
        pi_out = self._k_p * error + self.integrator

        # Calculate output phase
        self.output_phase += 2 * np.pi * pi_out * self._k_vco * (1 / SAMPLING_RATE)

        # Calculate base for sin and cos
        omega_prev = self.omega
        self.omega = 2 * np.pi * RECEIVER_CARRIER_FREQUENCY * sample_time + self.output_phase

        # Calculate current frequency
        self.frequency = ((self.omega - omega_prev) / (2 * np.pi)) * SAMPLING_RATE
        if self.frequency > 0:
            self.frequency_filtered = self.frequency_filtered * self._frequency_filter_k \
                                      + self.frequency * (1. - self._frequency_filter_k)

        # Calculate MOD (%) for ZCD
        omega_mod = int(self.omega % (np.pi * HALFCYCLES_PER_SYMBOL))

        # Zero-crossing @ HALFCYCLES_PER_SYMBOL * PI (symbol rate)
        if self._omega_mod_prev > 0 and omega_mod == 0:
            self._omega_mod_prev = omega_mod
            self.zero_crossing_flag = True
        self._omega_mod_prev = omega_mod

        # Calculate complex output
        vco_real = np.cos(self.omega)
        vco_imag = -np.sin(self.omega)
        self.vco = complex(vco_real, vco_imag)


def play_signal_and_record_output(signal: np.ndarray) -> np.ndarray:
    """
    Passes the signal through the audio interface using PyAudio
    :param signal: input data
    :return: output data (same size)
    """
    print("Importing PyAudio")
    import pyaudio

    # Buffer for recorded symbol
    recorded_signal = np.empty(1, dtype=np.float32)

    # Data type of received data from PyAudio
    dtype_ = np.dtype(np.float32)
    dtype_ = dtype_.newbyteorder("<")

    # Counter for chunks
    chunk_counter = 0

    # Flag for startign recording (if signal > CARRIER_LOST_AMPLITUDE)
    recording_started = False

    # Create an interface to PortAudio
    pyaudio_ = pyaudio.PyAudio()

    # Initialize stream
    stream = pyaudio_.open(format=pyaudio.paFloat32,
                           channels=1,
                           rate=SAMPLING_RATE,
                           frames_per_buffer=CHUNK_SIZE,
                           input=True,
                           output=True)

    # Start playing and recording
    print("Playback started")
    while True:
        # Stream input data
        if chunk_counter * CHUNK_SIZE + CHUNK_SIZE <= len(signal):
            playback_chunk = np.asarray(signal[chunk_counter * CHUNK_SIZE: chunk_counter * CHUNK_SIZE + CHUNK_SIZE],
                                        dtype=np.float32)
            stream.write(playback_chunk.tobytes("C"))

        # Stream zeros (to compensate delay)
        else:
            stream.write(np.zeros(CHUNK_SIZE, dtype=np.float32).tobytes("C"))

        # Read data from mic
        recorded_chunk_temp = stream.read(CHUNK_SIZE)
        recorded_chunk_temp = np.frombuffer(recorded_chunk_temp, dtype=dtype_)

        # Calculate chunk volume
        chunk_volume = 20 * np.log10(np.max(np.abs(recorded_chunk_temp)))

        # Log volume
        print("Current chunk volume: {:.1f} dBFS".format(chunk_volume))

        # Start recording
        if not recording_started and chunk_volume > CARRIER_START_AMPLITUDE:
            recording_started = True
            print("Recording started")

        # Append to recorded buffer
        if recording_started:
            recorded_signal = np.append(recorded_signal, recorded_chunk_temp)

        # Exit recording_started and no more data to record
        if recording_started and chunk_volume < CARRIER_LOST_AMPLITUDE:
            break

        # Exit if we exceeded maximum number of chunks to record
        if chunk_counter >= PLAYBACK_MAX_CHUNKS:
            print("Maximum number of chunks reached {} (NOTHING RECORDED). Aborting".format(PLAYBACK_MAX_CHUNKS))
            break

        # Increment number of chunks
        chunk_counter += 1

    # Close all stream
    stream.stop_stream()
    stream.close()
    pyaudio_.terminate()
    print("Recording and playback finished")

    # Trim to match input data size
    return recorded_signal[-len(signal):]


def main():
    # ############################# #
    # SIGNAL GENERATION (MODULATOR) #
    # ############################# #

    modulated_signal = np.zeros(1, dtype=np.float32)
    modulated_signal_t = np.zeros(1, dtype=np.float32)
    modulated_data_phase_plot = np.zeros(1, dtype=np.int8)
    modulated_data_i_plot = np.zeros(1, dtype=np.float32)
    modulated_data_q_plot = np.zeros(1, dtype=np.float32)
    modulated_data_constellation_plot = np.zeros((2, 1), dtype=np.float32)
    tx_data_position = 0
    symbol_samples_counter = 0
    value_i = 0.
    value_q = 0.
    phase = 0
    amplitude_correction_factor = 1. / np.sqrt(2.)
    while True:
        # Calculate next sample timestamp
        time_new = modulated_signal_t[-1] + 1. / SAMPLING_RATE

        if symbol_samples_counter == 0:
            # Check if we have data to modulate
            if tx_data_position < len(TX_DATA) - 1:
                # Increment counter
                tx_data_position += 1

                # Pop from list
                phase = TX_DATA[tx_data_position]
                if 0 <= phase <= 3:
                    value_i = 1. if phase & 0b10 == 0b10 else -1.
                    value_q = 1. if phase & 0b01 == 0b01 else -1.
                else:
                    value_i = 0.
                    value_q = 0.

                # "Play" new phase for some amount of cycles
                symbol_samples_counter = int((1. / TRANSMITTER_CARRIER_FREQUENCY / 2) * SAMPLING_RATE
                                             * HALFCYCLES_PER_SYMBOL)

            # Exit if no more data
            else:
                break

        # Decrement phase counter
        if symbol_samples_counter > 0:
            symbol_samples_counter -= 1

        # Generate wave
        sample_i = np.cos(2 * np.pi * TRANSMITTER_CARRIER_FREQUENCY * time_new) * value_i * amplitude_correction_factor
        sample_q = -np.sin(2 * np.pi * TRANSMITTER_CARRIER_FREQUENCY * time_new) * value_q * amplitude_correction_factor
        sample = sample_i + sample_q
        sample *= CARRIER_AMPLITUDE

        # Append time and generated sample to arrays
        modulated_signal = np.append(modulated_signal, sample)
        modulated_signal_t = np.append(modulated_signal_t, time_new)
        modulated_data_i_plot = np.append(modulated_data_i_plot, value_i)
        modulated_data_q_plot = np.append(modulated_data_q_plot, value_q)
        modulated_data_phase_plot = np.append(modulated_data_phase_plot, phase)
        modulated_data_constellation_plot = np.concatenate((modulated_data_constellation_plot,
                                                            [[value_i], [value_q]]),
                                                           axis=1)

    # Initialize filter
    modulation_filter = RRCFiler.RRCFilter(RRCFiler.FILTER_TYPE_BANDPASS,
                                           SAMPLING_RATE,
                                           TRANSMITTER_CARRIER_FREQUENCY - BANDWIDTH / 2,
                                           TRANSMITTER_CARRIER_FREQUENCY + BANDWIDTH / 2,
                                           positive_lobes_n=TRANSMITTER_FILTER_LOBES_N,
                                           alpha=TRANSMITTER_FILTER_ALPHA)

    # Compensate for RRC filter delay
    modulated_signal = np.append(modulated_signal, np.zeros(modulation_filter.filter_length // 2, dtype=np.float32))

    # Apply band-pass filter to isolate carrier
    for i in range(len(modulated_signal)):
        # Filter each sample
        modulated_signal[i] = modulation_filter.filter(modulated_signal[i])

        # Compensate amplitude
        modulated_signal[i] /= np.sqrt(2)

    # Compensate for RRC filter delay
    modulated_signal = modulated_signal[modulation_filter.filter_length // 2:]
    modulated_signal_t += modulation_filter.filter_length / 2 / SAMPLING_RATE

    # Print log info
    print("Generated {} samples. Total time: ~{:.3f}s".format(len(modulated_signal), modulated_signal_t[-1]))

    # ############################ #
    # TRANSMISSION LINE SIMULATION #
    # ############################ #

    if USE_LINE_SIMULATION:
        # Add delay
        delay_seconds = np.random.uniform(low=LINE_DELAY_FROM, high=LINE_DELAY_TO)
        delay_samples = int(delay_seconds * SAMPLING_RATE)
        modulated_signal = np.append(np.zeros(delay_samples, dtype=np.float32), modulated_signal)
        modulated_signal_t_from = modulated_signal_t[0]
        modulated_signal_t += delay_seconds
        modulated_signal_t_delay = np.arange(modulated_signal_t_from, modulated_signal_t[0], 1 / SAMPLING_RATE)[:-1]
        modulated_signal_t = np.append(modulated_signal_t_delay, modulated_signal_t)
        modulated_data_phase_plot = np.append(np.zeros(delay_samples, dtype=np.int8), modulated_data_phase_plot)
        modulated_data_i_plot = np.append(np.zeros(delay_samples, dtype=np.float32), modulated_data_i_plot)
        modulated_data_q_plot = np.append(np.zeros(delay_samples, dtype=np.float32), modulated_data_q_plot)
        print("Added ~{:.3f}s ({} samples) of delay".format(delay_seconds, delay_samples))

        # Copy
        transmitted_signal = modulated_signal.copy()

        # Add noise
        transmitted_signal += (np.random.rand(len(transmitted_signal)) - .5) * 2 * NOISE_LEVEL

        # Clip to -1 - 1 range
        transmitted_signal = np.clip(transmitted_signal, -1, 1)

        # Apply Low-pass filter
        line_filter = ButterworthFilter.Filter(ButterworthFilter.FILTER_TYPE_LOWPASS, SAMPLING_RATE, LINE_LFP_FREQUENCY,
                                               order=3)
        for i in range(len(transmitted_signal)):
            transmitted_signal[i] = line_filter.filter(transmitted_signal[i])

    # Real audio interface
    else:
        transmitted_signal = play_signal_and_record_output(modulated_signal)

    # ######################################## #
    # SIGNAL RECEIVING (DEMODULATOR + DECODER) #
    # ######################################## #

    # Filters
    receiver_filter = RRCFiler.RRCFilter(RRCFiler.FILTER_TYPE_BANDPASS,
                                         SAMPLING_RATE,
                                         RECEIVER_CARRIER_FREQUENCY - BANDWIDTH / 2,
                                         RECEIVER_CARRIER_FREQUENCY + BANDWIDTH / 2,
                                         positive_lobes_n=RECEIVER_FILTER_LOBES_N,
                                         alpha=RECEIVER_FILTER_ALPHA)
    filter_i = ButterworthFilter.Filter(ButterworthFilter.FILTER_TYPE_LOWPASS, SAMPLING_RATE,
                                        RECEIVER_CARRIER_FREQUENCY * 0.7, order=3)
    filter_q = ButterworthFilter.Filter(ButterworthFilter.FILTER_TYPE_LOWPASS, SAMPLING_RATE,
                                        RECEIVER_CARRIER_FREQUENCY * 0.7, order=3)

    # Compensate RRC filter delay
    transmitted_signal = np.append(transmitted_signal, np.zeros(receiver_filter.filter_length // 2, dtype=np.float32))

    # Initialize PLL class
    pll = DPLL(PLL_K_P, PLL_K_I, PLL_K_VCO)

    # Plottable data
    received_signal_t = np.empty(0, dtype=np.float32)
    received_signal_plot = np.empty(0, dtype=np.float32)
    received_signal_constellation_plot = np.empty((2, 1), dtype=np.float32)
    received_signal_agc_plot = np.empty(0, dtype=np.float32)
    pll_out_plot = np.empty(0, dtype=np.float32)
    i_mixed_plot = np.empty(0, dtype=np.float32)
    q_mixed_plot = np.empty(0, dtype=np.float32)
    zero_crossings_plot = np.empty(0, dtype=np.uint8)
    decoded_data_plot = np.empty(0, dtype=np.uint8)
    signs_i_plot = np.empty(0, dtype=np.int8)
    signs_q_plot = np.empty(0, dtype=np.int8)
    carrier_errors_plot = np.empty(0, dtype=np.float32)
    agc_gain_plot = np.empty(0, dtype=np.float32)
    sampling_points_plot = np.empty(0, dtype=np.uint8)
    decoded_data_constellation_plot = np.empty((2, 1), dtype=np.float32)

    # Decoded data
    decoded_data = []
    current_byte = 0
    current_byte_bits_position = 6

    # Automatic gain control variable
    agc_accumulator = 0.

    # Stores I and Q signs from previous cycle (for sampling detector)
    sign_i_prev = 0
    sign_q_prev = 0

    # Stores input of PLL
    carrier_error = -1

    # Will be True as soon as carrier is detected
    carrier_detected = False

    # Timer to measure PLL's stability (for locking)
    pll_locked_timer = 0.

    # Will be Trues as soon as
    pll_locked = False

    # Initial IQ phases for each symbol (0, 1, 2, 3)
    reference_signs_i = [0, 0, 0, 0]
    reference_signs_q = [0, 0, 0, 0]
    reference_signs_set_position = 0

    # Counter for samples after PLL's VCO's zero crossing (falling edge) (resets to 0 every falling edge)
    samples_after_zcd = 0

    # Where to sample I and Q values after samples_after_zcd (sampling at samples_after_zcd + sample_iq_position)
    sampling_position = -1

    # Record current time for benchmarking
    time_started = time.time()

    # Now we need to precess each sample individually
    for i in range(len(transmitted_signal)):
        # Extract sample
        received_sample = transmitted_signal[i]

        # Calculate current time in seconds
        current_time = i / SAMPLING_RATE

        # Filter current sample
        filtered_sample = receiver_filter.filter(received_sample)

        # Pass current sample to automatic gain control
        agc_input = filtered_sample

        # Calculate automatic gain control
        if abs(filtered_sample) > agc_accumulator:
            agc_accumulator = agc_accumulator * AGC_CONSTANT_RISING + abs(agc_input) * (1. - AGC_CONSTANT_RISING)
        else:
            agc_accumulator = agc_accumulator * AGC_CONSTANT_FALLING + abs(agc_input) * (1. - AGC_CONSTANT_FALLING)

        # Check if we have any AGC (also input signal)
        if agc_accumulator > 0.:
            # Set carrier detected flag
            signal_strength = 20 * np.log10(agc_accumulator)
            if not carrier_detected and signal_strength > CARRIER_START_AMPLITUDE:
                carrier_detected = True
                print("Carrier detected @ ~{:.3f}s".format(current_time))
            if carrier_detected and signal_strength < CARRIER_LOST_AMPLITUDE:
                carrier_detected = False
                pll_locked = False
                print("Carrier lost and PLL unlocked @ ~{:.3f}s".format(current_time))

            # Apply automatic gain
            if carrier_detected:
                filtered_sample *= AGC_CORRECTION_FACTOR / agc_accumulator

                # Hard-clip to -1 - 1 (to prevent huge AGC initial overshoot)
                filtered_sample = np.clip(filtered_sample, -1., 1.)

        # Mix input signal (real values only) with PLL output
        i_mixed = filtered_sample * pll.vco.imag
        q_mixed = filtered_sample * pll.vco.real

        # Apply LPFs
        i_mixed = filter_i.filter(i_mixed)
        q_mixed = filter_q.filter(q_mixed)

        # Calculate signs
        sign_i = 1 if i_mixed >= 0 else -1
        sign_q = 1 if q_mixed >= 0 else -1

        # Check if carrier is detected but PLL still not locked
        if carrier_detected and not pll_locked:
            # Start timer
            if pll_locked_timer == 0:
                pll_locked_timer = current_time

            # Reset timer on large error or first run
            if abs(carrier_error) > PLL_LOCKED_THRESHOLD:
                pll_locked_timer = 0

            # Check timer
            if pll_locked_timer > 0 and current_time - pll_locked_timer > PLL_LOCKED_TIME:
                # Reset timer
                pll_locked_timer = 0

                # Set variables
                pll_locked = True
                reference_signs_i[0] = sign_i
                reference_signs_q[0] = sign_q
                reference_signs_set_position += 1

                # Calculated PPM error
                carrier_error_hz = abs(RECEIVER_CARRIER_FREQUENCY - pll.frequency_filtered)
                carrier_error_ppm = (carrier_error_hz * 1e6) / RECEIVER_CARRIER_FREQUENCY

                print("PLL locked @ ~{:.3f}s. VCO: ~{:.2f}Hz "
                      "(error between {:.1f}Hz: {:.0f}ppm). I: {:.0f}, Q: {:.0f}".format(current_time,
                                                                                         pll.frequency_filtered,
                                                                                         RECEIVER_CARRIER_FREQUENCY,
                                                                                         carrier_error_ppm,
                                                                                         reference_signs_i[0],
                                                                                         reference_signs_q[0]))

        # Carrier not detected or PLL locked -> Reset timer
        else:
            pll_locked_timer = 0

        # Zero-crossing detector (falling edge) using PLL's carrier
        if pll.zero_crossing_flag:
            # Clear PLL's zero crossing flag
            pll.zero_crossing_flag = False

            # Reset samples counter
            samples_after_zcd = 0

            # Save data for plotting
            zero_crossings_plot = np.append(zero_crossings_plot, 1)

        # No zero-crossing detected
        else:
            # Increment samples counter
            samples_after_zcd += 1

            # Save data for plotting
            zero_crossings_plot = np.append(zero_crossings_plot, 0)

        # Calculate how many samples one symbol takes
        if pll.frequency_filtered > 0:
            one_symbol_samples = HALFCYCLES_PER_SYMBOL * (SAMPLING_RATE / pll.frequency_filtered / 2)
        else:
            one_symbol_samples = 0

        # Calculate sampling position
        if carrier_detected and pll_locked:
            # Calculate it only one time (at the first phase change)
            # I know that this is not very good, and we need to do a continues calculation (correction)
            if sampling_position < 0:
                # If sign of I or sign of Q changed (phase change)
                if (sign_i_prev >= 0 and sign_i < 0) or (sign_i_prev < 0 and sign_i >= 0) \
                        or (sign_q_prev >= 0 and sign_q < 0) or (sign_q_prev < 0 and sign_q >= 0):

                    # Calculate sampling position
                    sampling_position = samples_after_zcd + one_symbol_samples / 2
                    if sampling_position > one_symbol_samples:
                        sampling_position -= one_symbol_samples

                    if sampling_position > one_symbol_samples - 2:
                        sampling_position -= 2.

                    # Log phase change and sampling position
                    print("Detected phase change @ ~{:.3f}s. "
                          "Sampling position: {:.0f} samples after zero-cross".format(current_time,
                                                                                      sampling_position))

        # Reset sampling position
        else:
            sampling_position = -1

        # Sampling position is set -> Sample data
        if sampling_position >= 0 and samples_after_zcd == int(sampling_position):
            # Set other reference phases if 0 is set
            if 0 < reference_signs_set_position < len(reference_signs_i):
                # Check if sign changed
                if sign_i != reference_signs_i[reference_signs_set_position - 1] \
                        or sign_q != reference_signs_q[reference_signs_set_position - 1]:
                    reference_signs_i[reference_signs_set_position] = sign_i
                    reference_signs_q[reference_signs_set_position] = sign_q

                    print("IQ values of phase {} recorded @ ~{:.3f}s. I: {:.0f}, Q: {:.0f}"
                          .format(reference_signs_set_position, current_time, sign_i, sign_q))

                    reference_signs_set_position += 1

                # Save data for plotting
                sampling_points_plot = np.append(sampling_points_plot, 1)
                decoded_data_plot = np.append(decoded_data_plot, -1)
                decoded_data_constellation_plot = np.concatenate((decoded_data_constellation_plot, [[0], [0]]),
                                                                 axis=1)

            # Check if we have all reference values
            elif reference_signs_set_position >= len(reference_signs_i):
                # Decode data according to table of reference signs
                data_iq = 0
                for sign_n in range(4):
                    if reference_signs_i[sign_n] == sign_i and reference_signs_q[sign_n] == sign_q:
                        data_iq = sign_n
                        break

                # Append to list
                decoded_data.append(data_iq)
                print("Decoded data @ ~{:.3f}s: {} ({:#04b})".format(current_time, data_iq, data_iq))

                # Fill byte
                current_byte |= ((data_iq & 0b11) << current_byte_bits_position)
                current_byte_bits_position -= 2
                if current_byte_bits_position < 0:
                    # Log
                    print("Decoded byte: {:#010b}, char: {}".format(current_byte, chr(current_byte)))

                    # Reset variables
                    current_byte = 0
                    current_byte_bits_position = 6

                # Save data for plotting
                sampling_points_plot = np.append(sampling_points_plot, 1)
                decoded_data_plot = np.append(decoded_data_plot, data_iq)
                decoded_data_constellation_plot = np.concatenate((decoded_data_constellation_plot,
                                                                  [[i_mixed], [q_mixed]]),
                                                                 axis=1)

            # Ignore
            else:
                # Save data for plotting
                sampling_points_plot = np.append(sampling_points_plot, 1)
                decoded_data_plot = np.append(decoded_data_plot, -1)
                decoded_data_constellation_plot = np.concatenate((decoded_data_constellation_plot, [[0], [0]]),
                                                                 axis=1)
        # Ignore
        else:
            # Save data for plotting
            sampling_points_plot = np.append(sampling_points_plot, 0)
            decoded_data_plot = np.append(decoded_data_plot, -1)
            decoded_data_constellation_plot = np.concatenate((decoded_data_constellation_plot, [[0], [0]]), axis=1)

        # Save signs for next cycle
        sign_i_prev = sign_i
        sign_q_prev = sign_q

        # Reset signs on carrier lost
        if not pll_locked or not carrier_detected:
            reference_signs_set_position = 0
            for sign_n in range(len(reference_signs_i)):
                reference_signs_i[sign_n] = 0
                reference_signs_q[sign_n] = 0

        # Calculate PLL input
        pll_input_i = sign_i * q_mixed
        pll_input_q = sign_q * i_mixed

        # Calculate carrier error
        if carrier_detected:
            # Accept 4 possible cases ((I=1, Q=1), (I=-1, Q=1), (I=1, Q=-1), (I=-1, Q=-1))
            carrier_error = pll_input_q - pll_input_i
        else:
            carrier_error = 0.

        # Take one PLL step
        pll.step(carrier_error, current_time)

        # Save data for plotting
        received_signal_plot = np.append(received_signal_plot, received_sample)
        received_signal_t = np.append(received_signal_t, current_time)
        received_signal_agc_plot = np.append(received_signal_agc_plot, filtered_sample)
        agc_gain_plot = np.append(agc_gain_plot, agc_accumulator)
        pll_out_plot = np.append(pll_out_plot, pll.vco)
        i_mixed_plot = np.append(i_mixed_plot, i_mixed)
        q_mixed_plot = np.append(q_mixed_plot, q_mixed)
        signs_i_plot = np.append(signs_i_plot, sign_i)
        signs_q_plot = np.append(signs_q_plot, sign_q)
        carrier_errors_plot = np.append(carrier_errors_plot, carrier_error)
        received_signal_constellation_plot = np.concatenate((received_signal_constellation_plot,
                                                             [[i_mixed], [q_mixed]]),
                                                            axis=1)

    time_taken = time.time() - time_started
    print("Processing finished. Took {:.3f}s".format(time_taken))

    # ############# #
    # DATA PLOTTING #
    # ############# #
    # Compensate RRC filter delay
    zero_crossings_plot = zero_crossings_plot[receiver_filter.filter_length // 2:]
    sampling_points_plot = sampling_points_plot[receiver_filter.filter_length // 2:]
    decoded_data_plot = decoded_data_plot[receiver_filter.filter_length // 2:]
    received_signal_plot = received_signal_plot[receiver_filter.filter_length // 2:]
    received_signal_t = received_signal_t[receiver_filter.filter_length // 2:]
    received_signal_agc_plot = received_signal_agc_plot[receiver_filter.filter_length // 2:]
    agc_gain_plot = agc_gain_plot[receiver_filter.filter_length // 2:]
    pll_out_plot = pll_out_plot[receiver_filter.filter_length // 2:]
    i_mixed_plot = i_mixed_plot[receiver_filter.filter_length // 2:]
    q_mixed_plot = q_mixed_plot[receiver_filter.filter_length // 2:]
    signs_i_plot = signs_i_plot[receiver_filter.filter_length // 2:]
    signs_q_plot = signs_q_plot[receiver_filter.filter_length // 2:]
    carrier_errors_plot = carrier_errors_plot[receiver_filter.filter_length // 2:]

    # Calculate fft
    real_fft = np.fft.rfft(received_signal_agc_plot)
    s_mag = np.abs(real_fft) * 2 / (len(received_signal_agc_plot) / 2)
    min_value = np.finfo(np.float32).eps
    s_mag[s_mag < min_value] = min_value
    fft_dbfs = 20. * np.log10(s_mag)
    fft_freqs = np.fft.fftfreq(len(received_signal_agc_plot), d=1. / SAMPLING_RATE)
    fft_dbfs = fft_dbfs[:len(fft_dbfs) - 1]
    fft_freqs = fft_freqs[:len(fft_dbfs)]

    # Initialize matplotlib
    fig, axs = plt.subplots(3, 2, gridspec_kw={"width_ratios": [3, 0.75]})
    fig.set_figwidth(30)
    fig.set_figheight(15)
    fig.tight_layout(pad=3)

    # Encoded and decoded data
    fig.text(0.03, 0.95, "Encoded data: {}".format(", ".join(np.asarray(TX_DATA, dtype=str))), fontsize=12)
    fig.text(0.03, 0.30, "Decoded data: {}".format(", ".join(np.asarray(decoded_data, dtype=str))), fontsize=12)

    # Received signal FFT
    axs[0, 1].set_title("Received signal FFT (after filtering and AGC)")
    axs[0, 1].grid(True, which="both")
    axs[0, 1].set(xlabel="Frequency (Hz)", ylabel="Amplitude (dBFS)")
    axs[0, 1].set_xscale("log")
    axs[0, 1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs[0, 1].plot(fft_freqs, fft_dbfs, "tab:blue", label="Received signal FFT")

    # Received data continues constellation diagram
    axs[1, 1].set_title("Received data constellation diagram")
    axs[1, 1].grid(True, which="both")
    axs[1, 1].set_xlim((-1, 1))
    axs[1, 1].set_ylim((-1, 1))
    axs[1, 1].set_xlabel("I")
    axs[1, 1].set_ylabel("Q")
    plot_i, plot_q = np.split(received_signal_constellation_plot, 2, axis=0)
    axs[1, 1].scatter(plot_i, plot_q, c="tab:pink", label="Received signal", alpha=0.2)

    # Received data sampled constellation diagram
    axs[2, 1].set_title("Received and sampled data constellation diagram")
    axs[2, 1].grid(True, which="both")
    axs[2, 1].set_xlim((-1, 1))
    axs[2, 1].set_ylim((-1, 1))
    axs[2, 1].set_xlabel("I")
    axs[2, 1].set_ylabel("Q")
    plot_i, plot_q = np.split(decoded_data_constellation_plot, 2, axis=0)
    axs[2, 1].scatter(plot_i, plot_q, c="tab:orange", label="Decoded data", alpha=0.8)

    # Transmitter
    axs[0, 0].set_title("Transmitted data")
    axs[0, 0].grid(True, which="both")
    axs[0, 0].set(xlabel="Time (s)")
    axs[0, 0].plot(modulated_signal_t, modulated_data_phase_plot, "tab:blue", label="Input data")
    axs[0, 0].plot(modulated_signal_t, modulated_data_i_plot, "tab:olive", linestyle="dashed", label="I")
    axs[0, 0].plot(modulated_signal_t, modulated_data_q_plot, "tab:cyan", linestyle="dashed", label="Q")
    axs[0, 0].plot(modulated_signal_t, modulated_signal, "tab:orange", label="Modulated and filtered signal")

    # Receiver
    axs[1, 0].set_title("Received data")
    axs[1, 0].grid(True, which="both")
    axs[1, 0].set(xlabel="Time (s)")
    axs[1, 0].plot(received_signal_t, received_signal_plot, "tab:blue", label="Received signal", alpha=0.7)
    axs[1, 0].plot(received_signal_t, agc_gain_plot, "tab:orange", linestyle="dashed",
                   label="Automatic gain control (AGC)", alpha=0.7)
    axs[1, 0].plot(received_signal_t, received_signal_agc_plot, "tab:orange", label="Received signal after AGC",
                   alpha=0.7)
    axs[1, 0].plot(received_signal_t, i_mixed_plot, "tab:olive", label="I", alpha=0.7)
    axs[1, 0].plot(received_signal_t, q_mixed_plot, "tab:cyan", label="Q", alpha=0.7)
    axs[1, 0].plot(received_signal_t, carrier_errors_plot, "tab:red", label="PLL input (carrier error)", alpha=0.7)
    axs[1, 0].plot(received_signal_t, np.real(pll_out_plot), "tab:brown", label="PLL real output", alpha=0.7)

    # Receiver + demodulator
    axs[2, 0].set_title("Demodulated and decoded data")
    axs[2, 0].grid(True, which="both")
    axs[2, 0].set(xlabel="Time (s)")
    axs[2, 0].plot(received_signal_t, zero_crossings_plot, "tab:gray", label="PLL Zero-crossings", alpha=0.7)
    axs[2, 0].plot(received_signal_t, i_mixed_plot, "tab:olive", label="I", alpha=0.7)
    axs[2, 0].plot(received_signal_t, q_mixed_plot, "tab:cyan", label="Q", alpha=0.7)
    axs[2, 0].plot(received_signal_t, signs_i_plot, "tab:olive", linestyle="dashed", label="I sign", alpha=0.7)
    axs[2, 0].plot(received_signal_t, signs_q_plot, "tab:cyan", linestyle="dashed", label="Q sign", alpha=0.7)
    axs[2, 0].plot(received_signal_t, sampling_points_plot, "tab:pink", label="Sampling points", alpha=0.7)
    axs[2, 0].plot(received_signal_t, decoded_data_plot, "tab:orange", label="Decoded data", marker=".", alpha=0.7)

    # Show all plots
    fig.legend(loc="center left")
    fig.show()
    exit()


if __name__ == "__main__":
    main()
