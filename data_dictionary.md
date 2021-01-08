<a id='back_to_top'></a>

# Data Dictionary

<br><br>
List of Variables:<br>
- [filename](#filename)<br>
- [length](#length)<br>
- [chroma stft](#chroma)<br>
- [rms](#rms)<br>
- [spectral_centroid](#spectral_centroid)<br>
- [spectral_bandwidth](#spectral_bandwidth)<br>
- [rolloff](#rolloff)<br>
- [zero crossing rate](#zero_crossing)<br>
- [tempo](#tempo)<br>
- [mfcc](#mfcc)<br>
- [label](#label)<br>
<br>

- [summary table of variables](#summary_table)<br>

<a id='filename'></a>

---
**filename**<br><br>

Name of the file / observation, following the format: genre.number.wav (eg 'blues.00000.0.wav', 'hiphop.00045.3.wav', 'rock.00016.4.wav', etc)

[(back to top)](#back_to_top)

<a id='length'></a>

---
**length**<br><br>

Length of each track observation, given in sample number. All observations are the same value of 66149 samples; at a sample rate of 22050 Hz, this amounts to approx 3 second audio samples.

[(back to top)](#back_to_top)

<a id='chroma'></a>

---
**chroma_stft_mean**, **chroma_stft_var**<br><br>

Chroma represents the tonal content of an audio signal and is closely related to the pitch class. Short time fourier transforms are used to determine the sinusoidal frequency and phase content of local sections (ie short segments) of a signal as it changes over time.<br>
[Ref: Chroma feature extraction](https://www.researchgate.net/publication/330796993_Chroma_Feature_Extraction)<br>
[Ref: Short-time Fourier transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform#:~:text=The%20Short%2Dtime%20Fourier%20transform,as%20it%20changes%20over%20time.)

[(back to top)](#back_to_top)

<a id='rms'></a>

---
**rms_mean**, **rms_var**<br><br>

'Root mean square' is the average signal (or waveform) amplitude. Effectively, it is the average power of the signal. Root mean square is taken (as opposed to simply the arithmetic mean) in order to account for the negative sign that signals can take; signals oscillate between positive and negative.<br>
[Ref: Root mean square](https://en.wikipedia.org/wiki/Root_mean_square)

[(back to top)](#back_to_top)

<a id='spectral_centroid'></a>

---
**spectral_centroid_mean**, **spectral_centroid_var**<br><br>

The spectral centroid is the "centre of gravity" of the magnitude spectrum of the STFT:
<br><br>

<img src="https://render.githubusercontent.com/render/math?math=C_t = \frac{\sum_{n=1}^N M_t[n] * n}{\sum_{n=1}^N M_t[n]}">

<br>

where <img src="https://render.githubusercontent.com/render/math?math=M_t[n]"> is the magnitude of the Fourier transform at frame <img src="https://render.githubusercontent.com/render/math?math=t"> and frequency bin <img src="https://render.githubusercontent.com/render/math?math=n">. The centroid is a measure of spectral shape and higher centroid values correspond to “brighter” textures with more high frequencies.<br>
[Ref: Music genre classification (George Tzanetakis, 2002)](https://www.researchgate.net/publication/3333877_Musical_Genre_Classification_of_Audio_Signals)

[(back to top)](#back_to_top)

<a id='spectral_bandwidth'></a>

---
**spectral_bandwidth_mean**, **spectral_bandwidth_var**<br><br>

The spectral bandwidth describes the difference (bandwidth) between the highest and lowest frequencies (spectrum) present in the audio signal.<br> 
[Ref: Spectral bandwidth](https://www.andrew.cmu.edu/user/rk2x/telclass.dir/hw.dir/hw2s97sol.html)<br>
[Ref: Spectral width](https://en.wikipedia.org/wiki/Spectral_width)

[(back to top)](#back_to_top)

<a id='rolloff'></a>

---
**rolloff_mean**, **rolloff_var**<br><br>

The spectral rolloff is defined as the frequency below which 85% of the magnitude distribution is concentrated. The rolloff is another measure of spectral shape.<br>
[Ref: Music genre classification (George Tzanetakis, 2002)](https://www.researchgate.net/publication/3333877_Musical_Genre_Classification_of_Audio_Signals)

<img src="https://render.githubusercontent.com/render/math?math=\sum_{n=1}^{R_t} M_t [n] = 0.85 * \sum_{n=1}^{N} M_t [n]">

[(back to top)](#back_to_top)

<a id='zero_crossing'></a>

---
**zero_crossing_rate_mean**, **zero_crossing_rate_var**<br><br>

The zero crossing rate is the rate at which a signal changes from positive to zero to negative or from negative to zero to positive. Its value has been widely used in both speech recognition and music information retrieval, being a key feature to classify percussive sounds.<br>
[Ref: Zero-crossing rate](https://en.wikipedia.org/wiki/Zero-crossing_rate)

[(back to top)](#back_to_top)

<a id='tempo'></a>

---
**tempo**<br><br>

The tempo of the track, measured in beats per minute.

[(back to top)](#back_to_top)

<a id='mfcc'></a>

---
**mfcc**<br><br>

Mel Frequency Cepstral Coefficient (MFCC) are perceptually motivated features that are also based on the STFT (short time fourier transform). After taking the log-amplitude of the magnitude spectrum, the FFT bins are grouped and smoothed according to the perceptually motivated Mel-frequency scaling. Finally, in order to decorrelate the resulting feature vectors a discrete cosine transform is performed.<br>
[Ref: Music genre classification (George Tzanetakis, 2002)](https://www.researchgate.net/publication/3333877_Musical_Genre_Classification_of_Audio_Signals)
<br><br>
Derivation of MFCC:<br>1. Take the Fourier transform of (a windowed excerpt of) a signal.<br>2. Map the powers of the spectrum obtained above onto the mel scale, using triangular overlapping windows.<br>3. Take the logs of the powers at each of the mel frequencies.<br>4. Take the discrete cosine transform of the list of mel log powers, as if it were a signal.<br>5. The MFCCs are the amplitudes of the resulting spectrum.<br>
[Ref: Mel-frequency cepstrum](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
<br><br>
Converting frequency scale to mel scale:<br>

<img src="https://render.githubusercontent.com/render/math?math=m = 2595 . log(1 plus \frac{f}{700})">

Traditionally, the first 13 coefficients are taken as they contain information about the formants. These 13 coefficients are commonly used for speech recognition.<br>
Taking the first and second derivatives (<img src="https://render.githubusercontent.com/render/math?math=\Delta mfcc">, and <img src="https://render.githubusercontent.com/render/math?math=\Delta\Delta mfcc">) provides further coefficients (up to 26 and 39 respectively), which assist to boost accuracy in modelling.<br><br>

[(back to top)](#back_to_top)

<a id='label'></a>

---
**label**<br><br>

The genre of the track observation ('blues', 'hiphop', 'rock', 'classical', etc). 10 unique labels for the 10 different genres.<br> 
This is the target (dependent) variable.

[(back to top)](#back_to_top)

<a id='summary_table'></a>

---

|Variable | Description |
| :- | :- |
| **filename**<br><br> | Name of the file / observation, following format: genre.number.wav (eg 'blues.00000.0.wav', 'hiphop.00045.3.wav', 'rock.00016.4.wav', etc<br><br><br> |
| **length**<br><br> | Length of each track observation, given in sample number. All observations are the same value of 66149 samples; at a sample rate of 22050 Hz, this amounts to approx 3 second audio samples.<br><br><br>|
| **chroma_stft_mean**<br>**chroma_stft_var**<br><br> | Chroma represents the tonal content of an audio signal and is closely related to the pitch class. Short time fourier transforms are used to determine the sinusoidal frequency and phase content of local sections (ie short segments) of a signal as it changes over time.<br>[Ref: Chroma feature extraction](https://www.researchgate.net/publication/330796993_Chroma_Feature_Extraction)<br>[Ref: Short-time Fourier transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform#:~:text=The%20Short%2Dtime%20Fourier%20transform,as%20it%20changes%20over%20time.)<br><br><br>|
| **spectral_centroid_mean**<br>**spectral_centroid_var** | The spectral centroid is the "centre of gravity" of the magnitude spectrum of the STFT:<br><br><img src="https://render.githubusercontent.com/render/math?math=C_t = \frac{\sum_{n=1}^N M_t[n] * n}{\sum_{n=1}^N M_t[n]}"><br><br>where <img src="https://render.githubusercontent.com/render/math?math=M_t[n]"> is the magnitude of the Fourier transform at frame <img src="https://render.githubusercontent.com/render/math?math=t"> and frequency bin <img src="https://render.githubusercontent.com/render/math?math=n">. The centroid is a measure of spectral shape and higher centroid values correspond to “brighter” textures with more high frequencies.<br>[Ref: Music genre classification (George Tzanetakis, 2002)](https://www.researchgate.net/publication/3333877_Musical_Genre_Classification_of_Audio_Signals)
| **spectral_bandwidth_mean**<br>**spectral_bandwidth_var** | The spectral bandwidth describes the difference (bandwidth) between the highest and lowest frequencies (spectrum) present in the audio signal.<br>[Ref: Spectral bandwidth](https://www.andrew.cmu.edu/user/rk2x/telclass.dir/hw.dir/hw2s97sol.html)<br>[Ref: Spectral width](https://en.wikipedia.org/wiki/Spectral_width)<br><br><br>|
| **rolloff_mean**<br>**rolloff_var** | The spectral rolloff is defined as the frequency below which 85% of the magnitude distribution is concentrated. The rolloff is another measure of spectral shape.<br>[Ref: Music genre classification (George Tzanetakis, 2002)](https://www.researchgate.net/publication/3333877_Musical_Genre_Classification_of_Audio_Signals)<br><br><img src="https://render.githubusercontent.com/render/math?math=\sum_{n=1}^{R_t} M_t [n] = 0.85 * \sum_{n=1}^{N} M_t [n]"><br><br><br>|
| **zero_crossing_rate_mean**<br>**zero_crossing_rate_var** | The zero crossing rate is the rate at which a signal changes from positive to zero to negative or from negative to zero to positive. Its value has been widely used in both speech recognition and music information retrieval, being a key feature to classify percussive sounds.<br>[Ref: Zero-crossing rate](https://en.wikipedia.org/wiki/Zero-crossing_rate)<br><br><br>|
| **tempo**<br><br> | The tempo of the track, measured in beats per minute.<br><br><br> |
| **mfcc**<br><br> | Mel Frequency Cepstral Coefficient (MFCC) are perceptually motivated features that are also based on the STFT (short time fourier transform). After taking the log-amplitude of the magnitude spectrum, the FFT bins are grouped and smoothed according to the perceptually motivated Mel-frequency scaling. Finally, in order to decorrelate the resulting feature vectors a discrete cosine transform is performed.<br>[Ref: Music genre classification (George Tzanetakis, 2002)](https://www.researchgate.net/publication/3333877_Musical_Genre_Classification_of_Audio_Signals)<br><br>Derivation of MFCC:<br>1. Take the Fourier transform of (a windowed excerpt of) a signal.<br>2. Map the powers of the spectrum obtained above onto the mel scale, using triangular overlapping windows.<br>3. Take the logs of the powers at each of the mel frequencies.<br>4. Take the discrete cosine transform of the list of mel log powers, as if it were a signal.<br>5. The MFCCs are the amplitudes of the resulting spectrum.<br>[Ref: Mel-frequency cepstrum](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)<br><br>Converting frequency scale to mel scale:<br><img src="https://render.githubusercontent.com/render/math?math=m = 2595 . log(1 plus \frac{f}{700})"><br><br>Traditionally, the first 13 coefficients are taken as they contain information about the formants. These 13 coefficients are commonly used for speech recognition.<br>Taking the first and second derivatives (<img src="https://render.githubusercontent.com/render/math?math=\Delta mfcc">, and <img src="https://render.githubusercontent.com/render/math?math=\Delta\Delta mfcc">) provides further coefficients (up to 26 and 39 respectively), which assist to boost accuracy in modelling.<br><br><br> |
| **label**<br><br> | The genre of the track observation ('blues', 'hiphop', 'rock', 'classical', etc). 10 unique labels for the 10 different genres.<br>This is the target (dependent) variable.<br><br><br> |

[(back to top)](#back_to_top)
