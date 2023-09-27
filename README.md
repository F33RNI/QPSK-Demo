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
Generated 1729 samples. Total time: ~0.216s
Added ~0.000s (1 samples) of delay
Carrier detected @ ~0.004s
PLL locked @ ~0.099s. VCO: ~999.69Hz (error between 950.0Hz: 52309ppm). I: 1, Q: -1
Detected phase change @ ~0.124s. Sampling position: 5 samples after zero-cross
IQ values of phase 1 recorded @ ~0.125s. I: 1, Q: 1
IQ values of phase 2 recorded @ ~0.127s. I: -1, Q: -1
IQ values of phase 3 recorded @ ~0.129s. I: -1, Q: 1
Decoded data @ ~0.131s: 1 (0b01)
Decoded data @ ~0.133s: 0 (0b00)
Decoded data @ ~0.135s: 2 (0b10)
Decoded data @ ~0.137s: 0 (0b00)
Decoded byte: 0b01001000, char: H
Decoded data @ ~0.139s: 1 (0b01)
Decoded data @ ~0.141s: 2 (0b10)
Decoded data @ ~0.143s: 1 (0b01)
Decoded data @ ~0.145s: 1 (0b01)
Decoded byte: 0b01100101, char: e
Decoded data @ ~0.147s: 1 (0b01)
Decoded data @ ~0.149s: 2 (0b10)
Decoded data @ ~0.151s: 3 (0b11)
Decoded data @ ~0.153s: 0 (0b00)
Decoded byte: 0b01101100, char: l
Decoded data @ ~0.155s: 1 (0b01)
Decoded data @ ~0.157s: 2 (0b10)
Decoded data @ ~0.159s: 3 (0b11)
Decoded data @ ~0.161s: 0 (0b00)
Decoded byte: 0b01101100, char: l
Decoded data @ ~0.163s: 1 (0b01)
Decoded data @ ~0.165s: 2 (0b10)
Decoded data @ ~0.167s: 3 (0b11)
Decoded data @ ~0.169s: 3 (0b11)
Decoded byte: 0b01101111, char: o
Decoded data @ ~0.171s: 0 (0b00)
Decoded data @ ~0.173s: 2 (0b10)
Decoded data @ ~0.175s: 0 (0b00)
Decoded data @ ~0.177s: 1 (0b01)
Decoded byte: 0b00100001, char: !
Decoded data @ ~0.179s: 2 (0b10)
Decoded data @ ~0.181s: 1 (0b01)
Decoded data @ ~0.183s: 3 (0b11)
Decoded data @ ~0.185s: 0 (0b00)
Decoded byte: 0b10011100, char: 
Decoded data @ ~0.187s: 1 (0b01)
Decoded data @ ~0.189s: 2 (0b10)
Decoded data @ ~0.191s: 1 (0b01)
Decoded data @ ~0.193s: 1 (0b01)
Decoded byte: 0b01100101, char: e
Decoded data @ ~0.195s: 0 (0b00)
Carrier lost and PLL unlocked @ ~0.196s
Processing finished. Took 0.246s
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
