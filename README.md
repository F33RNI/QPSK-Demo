# 📡 QPSK-Demo
### Demonstration of a QPSK modulator / demodulator in the audio range (with the ability to run it through a real speaker / mic)

![](plot.png)

----------

### Block diagram of modulator section:

![](drawio/modulator.png)

### Block diagram of demodulator section:

![](drawio/demodulator.png)

As you can see, I'm using a Costas Loop with a PI (Proportional-Integrator controller) to control the local oscillator

----------

The screenshot at the top shows an example of a "Hello!" message.

`TX_DATA`:
```python
# [-1] - Silence,
# [0] * 60 - PLL sync burst,
# [0, 1, 2, 3] - phase (IQ) sync (MUST BE EXACTLY THESE VALUES IN THIS ORDER),
# string_to_qpsk_array("Hello!") - Data (payload),
# [-1] * 20 - Silence
TX_DATA = [-1] + [0] * 60 + [0, 1, 2, 3] + string_to_qpsk_array("Hello!") + [-1] * 20
```

Program output:
```
Generated 1729 samples. Total time: ~0.236s
Added ~0.007s (56 samples) of delay
Carrier detected @ ~0.029s
PLL locked @ ~0.118s. VCO: ~999.64Hz (error between 950.0Hz: 52254ppm). I: 1, Q: 1
Detected phase change @ ~0.150s. Sampling position: 13 samples after zero-cross
IQ values of phase 1 recorded @ ~0.151s. I: -1, Q: 1
IQ values of phase 2 recorded @ ~0.153s. I: 1, Q: -1
IQ values of phase 3 recorded @ ~0.155s. I: -1, Q: -1
Decoded data @ ~0.157s: 1 (0b01)
Decoded data @ ~0.159s: 0 (0b00)
Decoded data @ ~0.161s: 2 (0b10)
Decoded data @ ~0.163s: 0 (0b00)
Decoded byte: 0b01001000, char: H
Decoded data @ ~0.165s: 1 (0b01)
Decoded data @ ~0.167s: 2 (0b10)
Decoded data @ ~0.169s: 1 (0b01)
Decoded data @ ~0.171s: 1 (0b01)
Decoded byte: 0b01100101, char: e
Decoded data @ ~0.173s: 1 (0b01)
Decoded data @ ~0.175s: 2 (0b10)
Decoded data @ ~0.177s: 3 (0b11)
Decoded data @ ~0.179s: 0 (0b00)
Decoded byte: 0b01101100, char: l
Decoded data @ ~0.181s: 1 (0b01)
Decoded data @ ~0.183s: 2 (0b10)
Decoded data @ ~0.185s: 3 (0b11)
Decoded data @ ~0.187s: 0 (0b00)
Decoded byte: 0b01101100, char: l
Decoded data @ ~0.189s: 1 (0b01)
Decoded data @ ~0.191s: 2 (0b10)
Decoded data @ ~0.193s: 3 (0b11)
Decoded data @ ~0.195s: 3 (0b11)
Decoded byte: 0b01101111, char: o
Decoded data @ ~0.197s: 0 (0b00)
Decoded data @ ~0.199s: 2 (0b10)
Decoded data @ ~0.201s: 0 (0b00)
Decoded data @ ~0.203s: 1 (0b01)
Decoded byte: 0b00100001, char: !
Decoded data @ ~0.205s: 2 (0b10)
Decoded data @ ~0.207s: 2 (0b10)
Decoded data @ ~0.209s: 0 (0b00)
Decoded data @ ~0.211s: 0 (0b00)
Decoded byte: 0b10100000, char:  
Decoded data @ ~0.213s: 3 (0b11)
Decoded data @ ~0.215s: 1 (0b01)
Decoded data @ ~0.217s: 3 (0b11)
Decoded data @ ~0.219s: 2 (0b10)
Decoded byte: 0b11011110, char: Þ
Carrier lost and PLL unlocked @ ~0.220s
Processing finished. Took 0.277s
```

As you can see the message is received and decoded successfully:
```
Decoded byte: 0b01001000, char: H
Decoded byte: 0b01100101, char: e
Decoded byte: 0b01101100, char: l
Decoded byte: 0b01101100, char: l
Decoded byte: 0b01101111, char: o
Decoded byte: 0b00100001, char: !
```

----------

### To test transmission using your speaker and microphone:

1. Install the PyAudio package
2. Ensure that your sound card supports the float32 format and the values of `CHUNK_SIZE` and `SAMPLING_RATE`
3. Set `USE_LINE_SIMULATION` to `False`
4. Make sure your speaker and microphone are selected as the default devices in your system
5. Place the microphone near the speaker and create a quiet environment
6. Run the script. You should hear the transmission. If the sound level is set correctly, you will see a message indicating the start and end of recording
