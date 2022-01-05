#Direção pseye
#%% Import

import pytta 
import wave 
import arlpy.bf as bf
import pyaudio
import librosa
from pseye_device_index import device_index
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.io.wavfile as wav
import scipy.fft 
from signaltonoise import SNR, SNR_dB



#%% Gravação do sinal 

session = pyaudio.PyAudio() #iniciando a sessão

#definindo variáveis fixas
CHUNK = 1024  
FORMAT = pyaudio.paInt16 #16 bits
NCHANNELS = 4
RATE = 16000
DEVICE_INDEX = device_index(session) #procurando e definindo o PSeye como input
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"
MAX = 2**(15) #>> 2^(bits-1): máxima resolução

#iniciando o stream 
stream = session.open(format=FORMAT,
                      channels=NCHANNELS,
                      rate=RATE,
                      input=True,
                      input_device_index=DEVICE_INDEX,
                      frames_per_buffer=CHUNK)


print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
    #print(i)

print("* done recording")

stream.stop_stream()    
stream.close()
session.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(NCHANNELS)
wf.setsampwidth(session.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

#%% #################################
# Opening a wave file as numpy array
#####################################

rate,data = wav.read('output.wav') #scipy.io.wavefile as wav
timeVector = np.linspace(0., data.shape[0]/rate, data.shape[0]) #timeVector vector

ch1 = data[:, 0]
ch2 = data[:, 1]
ch3 = data[:, 2]
ch4 = data[:, 3]

ch_sum = (ch1+ch2+ch3+ch4)/4 # ()-> sum first and then divide by 4

wav.write('channels_sum.wav', rate, ch_sum.astype(np.int16)) #writing wav

#%% ###############################
# ATRASAR E SOMAR (delay_and_sum) 
###################################
# quando angles = 0,181,366 => 5 sequncias de 36 valores iguais.
# quando angles = 0,181,5 => 5 valores diferentes iguais aos repetidos anteriormente.
# :: 0,366,10
# mesmo com o angulo ~30 graus, o maior valor rms aponta pra 90.
# com 365 aponta quase para 135, coerente com a posição medida;
# proximidade com a fonte? sinal utilizado: voz mesmo; ruído também td certo0o
# 90 graus é a frente do pseye. 10 angulos observados ja é o suficiente para
# "apontar", mas não *localizar*. Não é preciso mas é o "melhor chute".

c = 345 # vel prog. som no ar [m/s]
positions = np.array([-0.04, -0.02, 0.02, 0.04]) #sensors positions (array)
angles = np.deg2rad(np.linspace(0,366,10)) #ângulos de varredura

steering = bf.steering_plane_wave(positions, c, angles)
DS = bf.delay_and_sum(np.array([ch1,ch2,ch3,ch4]), 16000, steering)

##%%
#cortando o sinal para calcular melhor o rms 
# (às vezes tem um grande pico no incio)
# cortar o sinal facilita o foco em um sinal tão complexo quanto a voz
# Amplitude usando MAX = 2**15 ~ 0.3(pico)
# amplitude usando max sinal = 1 ok ok alles gut

rms = []
for dir in DS:
    d = dir[15000:65000]/(max(abs(dir[15000:65000])))
    rms.append(np.sqrt(np.mean(d**2)))
# return index best_guess

##%% ########################
# POLAR PLOT: RMS x direction  
#############################

RMS=[]
[RMS.append(10*np.log10(r)) for r in rms]
    
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(angles, RMS)
#ax.set_rmax(1)
#ax.set_rticks(np.arange(0, 0.1)) # radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)                                                                                                                    

ax.set_title("Recognition RMS[dB] for each direction", va='bottom')

#%% delay and sum vs. channel 2 vs soma simples 
###############################################
# ch2 => np.argmax(ch2)=len(DS[0])+2

d = np.pad(DS[2], (2,2), 'constant', constant_values=(0,0)) #2 zeros no inicio e 2 no final

D = scipy.fft.fft(d[15000:65000]) #delay and sum
CS = scipy.fft.fft(ch_sum[15000:65000]) # soma simples
CH2 = scipy.fft.fft(ch2[15000:65000]) #channel 2 

plt.semilogx(10*np.log10(abs(D)), label='Delay and sum')
plt.semilogx(10*np.log10(abs(CS)), label='soma simples')
plt.semilogx(10*np.log10(abs(CH2)), label='Canal 2')
plt.xlim([0.0, 16000.0])
plt.xlabel("Frequência [Hz]")
plt.ylabel("Amplitude")
#plt.xlim([-0.01,0.1])
plt.legend()

s1 = SNR_dB(abs(D))
s2 = SNR_dB(abs(CH2))

s1/s2 # np.exp(1.84)= 6.2

#%% Ganho do Array G = 10log10(M)

d_norm = d[15000:65000]/max(abs(d[15000:65000]))
ch2_norm = ch2[15000:65000]/max(abs(d[15000:65000]))

snr_dir = SNR(d_norm)
snr_ch2 = SNR(ch2_norm)

arrayGain = snr_dir/snr_ch2 # ~-5.80

arrayGain

plt.plot(d_norm, label='delay and sum')
plt.plot(ch2_norm,label='canal 2')
plt.legend()
# %%
# The noise strongly affects the recognizer, making it impossible to detect voice in 
# a signal with "too long" moments of silence (background noise). Therefore, we must 
# window the signal so the recognizer can focus on the voice and then be able to
# recognize it properly. While the full signal gets null output from the recognizer, 
# the windowed signal is easily recognized, some with a high confidence score.
#

import speech_recognition as sr

r = sr.Recognizer()

filename = 'atraso_soma.wav'
wav.write(filename, rate, ch2_norm[15000:65000].astype(np.int16))

audio_file = sr.AudioFile(filename)

with audio_file as source:
    audio_data = r.record(source)

print(r.recognize_google(audio_data, language='en-US', show_all=True))







########## TO DO : GANHO DO ARRAY
#%% PLOT: sinal no tempo - DS[0] vs. ch_sum; 

# não exatamente sincronizado: DS[0].shape=47100; chm_sum=47104;
# len(ch_sum) = len(DS[0].shape) + 4
# índice do ponto máximo de cada sinal:
# np.argmax(DS[0]) = np.argmax(ch_sum - 2)
# ch_sum está atrasado 2 amostras; já que são 4 amostras de diferença:
# colocar 2 zeros no início de DS[0] e 2 zeros no final;
# try:
#assert np.argmax(d) == np.argmax(ch_sum)
# except: aSSERTIONError não sincronizados fia

d = np.pad(DS[0], (2,2), 'constant', constant_values=(0,0)) #2 zeros no inicio e 2 no final
ds_max = max(abs(DS[0]))

plt.plot(timeVector, d/ds_max, label='delay and sum')
plt.plot(timeVector, ch_sum/ds_max, label='soma simples')
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
#plt.xlim([-0.01,0.1])
plt.legend()


