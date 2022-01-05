import io
import wave
import scipy.io.wavfile as wavfile
import arlpy.bf as bf 
import audioop
import collections
import pyaudio 
import numpy as np
import math
import os.path 
from speech_recognition import AudioSource, AudioData, Recognizer, WaitTimeoutError


class PSMic(AudioSource):
    """
    PS Eye microphone class: sr.Microphone()'s evil twin.
    """
    def __init__(self,
                 nChannels=4,
                 samplingRate=16000,
                 chunkSize=1024,
                 device_index=None,
                 audioFormat=pyaudio.paInt16):    
        assert device_index is None or isinstance(device_index, int), "Device index must be None or an integer"
        assert samplingRate is None or (isinstance(samplingRate, int) and samplingRate > 0), "Sample rate must be None or a positive integer"
        assert isinstance(chunkSize, int) and chunkSize > 0, "Chunk size must be a positive integer"
    
        self.nChannels = nChannels
        self.SAMPLE_RATE = samplingRate
        self.CHUNK = chunkSize
        self.audioFormat = audioFormat
        self.session = pyaudio.PyAudio() #pyaudio instance
        self.stream = None
        self.device_index = self.device_index()
        self.SAMPLE_WIDTH = pyaudio.get_sample_size(self.audioFormat)  # size of each sample (16 bits = 2 bytes)


    def __enter__(self):
        assert self.stream is None, "This audio source is already inside a context manager"

        try:
            self.stream = PSMic.MicStream(
                self.session.open(
                    format=self.audioFormat,
                    channels=self.nChannels,
                    rate=self.SAMPLE_RATE,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.CHUNK
                    )
                    
            )
        except Exception:
            self.session.terminate()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.stream.close()
        finally:
            self.stream = None
            self.session.terminate()

    class MicStream(object):
        def __init__(self, pyaudio_stream):
            self.pyaudio_stream = pyaudio_stream
            #self.pos = [-0.04, -0.02, 0.02, 0.04] #sum(Xm)=0 (soma das posições = 0 (origem e centro do phase array))

        def read(self, size):
            """ definindo"""

            return self.pyaudio_stream.read(size, exception_on_overflow=False)

        def close(self):
            try:
                # sometimes, if the stream isn't stopped, closing the stream throws an exception
                if not self.pyaudio_stream.is_stopped():
                    self.pyaudio_stream.stop_stream()
            finally:
                self.pyaudio_stream.close()


    def list_microphone_names(self): 
        """
        List input by name
        """
        mic_list=[]
        for i in range(self.session.get_device_count()):
            mic_list.append(self.session.get_device_info_by_index(i)['name'])           
        return mic_list


    def device_index(self): #rewrite as static method
        """
        Function to detect PS Eye mic input and get PS Eye index.
        
        Returns device_index 
        """
        CAM = 'USB Camera-B4.04.27.1' #say my name
        mic_list=self.list_microphone_names() 

        try:            
            device_index = [CAM in item for item in mic_list].index(True) 
            #print(mic_list[device_index])
            return device_index
        except ValueError:
            print("Are you sure PS Eye is connected?")
            raise 




class PSRecognizer(Recognizer): 
    """
    PSRecognizer(Recognizer) rewrites listen() function adding delay_sum_best_guess(), 
    who chooses the best direction before outputting the audio signal.

    """

    def best_guess(self, DS, sample_width):
        rms = []
        for direction in DS: 
            rms.append(audioop.rms(direction, sample_width)) #rms.append((np.mean(direction**2))**0.5)
        chosen = DS[np.argmax(rms)]
        return chosen

    def delay_sum_best_guess(self, frame_data, angle_range, nChannels, sample_rate=16000, sample_width=2):

        # generate the WAV file contents
        with io.BytesIO() as wav_file:
            wav_writer = wave.open(wav_file, "wb")
            try:  # note that we can't use context manager, since that was only added in Python 3.4
                wav_writer.setframerate(sample_rate)#16000)#source.SAMPLE_RATE)
                wav_writer.setsampwidth(sample_width)#source.SAMPLE_WIDTH)
                wav_writer.setnchannels(nChannels)
                wav_writer.writeframes(frame_data)
                wav_data = wav_file.getvalue()
            finally:  # make sure resources are cleaned up
                wav_writer.close()
    
        sr,wavdata = wavfile.read(io.BytesIO(wav_data)) #(wavfile>scipy)

        if nChannels != 1:        
            channels = []
            [channels.append(wavdata[:,ch]) for ch in range(nChannels)] #reorganizing channels
            
            DS = bf.delay_and_sum(np.array(channels),
                                  sample_rate,
                                  self.steering(angle_range,nSensors=nChannels)) #delay and sum; 
            
            chosen = self.best_guess(DS, sample_width) #best guess
        else: chosen=wavdata;
        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        wavfile.write(byte_io, sample_rate, chosen.astype(np.int16))
        frame_data_ds = byte_io.read() # back to bytesss
        
        return frame_data_ds

    @staticmethod #static method for now
    def steering(angle_range, nSensors):

        c = 345 # vel prog. som no ar [m/s]
        start, end, step = angle_range  
        angles = np.deg2rad(np.arange(start,end,step))#np.linspace(start,end,step)) #ângulos de varredura
        positions = np.array([-0.04, -0.02, 0.02, 0.04]) #sensors positions (array)
        steering = bf.steering_plane_wave(positions[0:nSensors], c, angles)

        return steering


    def listen(self, source, angle_range=[0,366,10], timeout=None, phrase_time_limit=None, snowboy_configuration=None):
        """
        Records a single phrase from ``source`` (an ``AudioSource`` instance) into an ``AudioData`` instance, which it returns.
        This is done by waiting until the audio has an energy above ``recognizer_instance.energy_threshold`` (the user has started speaking), and then recording until it encounters ``recognizer_instance.pause_threshold`` seconds of non-speaking or there is no more audio input. The ending silence is not included.
        The ``timeout`` parameter is the maximum number of seconds that this will wait for a phrase to start before giving up and throwing an ``speech_recognition.WaitTimeoutError`` exception. If ``timeout`` is ``None``, there will be no wait timeout.
        The ``phrase_time_limit`` parameter is the maximum number of seconds that this will allow a phrase to continue before stopping and returning the part of the phrase processed before the time limit was reached. The resulting audio will be the phrase cut off at the time limit. If ``phrase_timeout`` is ``None``, there will be no phrase time limit.
        The ``snowboy_configuration`` parameter allows integration with `Snowboy <https://snowboy.kitt.ai/>`__, an offline, high-accuracy, power-efficient hotword recognition engine. When used, this function will pause until Snowboy detects a hotword, after which it will unpause. This parameter should either be ``None`` to turn off Snowboy support, or a tuple of the form ``(SNOWBOY_LOCATION, LIST_OF_HOT_WORD_FILES)``, where ``SNOWBOY_LOCATION`` is the path to the Snowboy root directory, and ``LIST_OF_HOT_WORD_FILES`` is a list of paths to Snowboy hotword configuration files (`*.pmdl` or `*.umdl` format).
        This operation will always complete within ``timeout + phrase_timeout`` seconds if both are numbers, either by returning the audio data, or by raising a ``speech_recognition.WaitTimeoutError`` exception.
        """
        assert isinstance(source, AudioSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before listening, see documentation for ``AudioSource``; are you using ``source`` outside of a ``with`` statement?"
        assert self.pause_threshold >= self.non_speaking_duration >= 0
        if snowboy_configuration is not None:
            assert os.path.isfile(os.path.join(snowboy_configuration[0], "snowboydetect.py")), "``snowboy_configuration[0]`` must be a Snowboy root directory containing ``snowboydetect.py``"
            for hot_word_file in snowboy_configuration[1]:
                assert os.path.isfile(hot_word_file), "``snowboy_configuration[1]`` must be a list of Snowboy hot word configuration files"

        seconds_per_buffer = float(source.CHUNK) / source.SAMPLE_RATE
        pause_buffer_count = int(math.ceil(self.pause_threshold / seconds_per_buffer))  # number of buffers of non-speaking audio during a phrase, before the phrase should be considered complete
        phrase_buffer_count = int(math.ceil(self.phrase_threshold / seconds_per_buffer))  # minimum number of buffers of speaking audio before we consider the speaking audio a phrase
        non_speaking_buffer_count = int(math.ceil(self.non_speaking_duration / seconds_per_buffer))  # maximum number of buffers of non-speaking audio to retain before and after a phrase

        # read audio input for phrases until there is a phrase that is long enough
        elapsed_time = 0  # number of seconds of audio read
        buffer = b""  # an empty buffer means that the stream has ended and there is no data left to read
        while True:
            frames = collections.deque()

            if snowboy_configuration is None:
                # store audio input until the phrase starts
                while True:
                    # handle waiting too long for phrase by raising an exception
                    elapsed_time += seconds_per_buffer
                    if timeout and elapsed_time > timeout:
                        raise WaitTimeoutError("listening timed out while waiting for phrase to start")

                    buffer = source.stream.read(source.CHUNK) # le o buffer pela 1 vez 
                    if len(buffer) == 0: break  # reached end of the stream
                    frames.append(buffer)
                    if len(frames) > non_speaking_buffer_count:  # ensure we only keep the needed amount of non-speaking buffers
                        frames.popleft()

                    # detect whether speaking has started on audio input
                    energy = audioop.rms(buffer, source.SAMPLE_WIDTH)  # energy of the audio signal
                    if energy > self.energy_threshold: break

                    # dynamically adjust the energy threshold using asymmetric weighted average
                    if self.dynamic_energy_threshold:
                        damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer  # account for different chunk sizes and rates
                        target_energy = energy * self.dynamic_energy_ratio
                        self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)
            else:
                # read audio input until the hotword is said
                snowboy_location, snowboy_hot_word_files = snowboy_configuration
                buffer, delta_time = self.snowboy_wait_for_hot_word(snowboy_location, snowboy_hot_word_files, source, timeout)
                elapsed_time += delta_time
                if len(buffer) == 0: break  # reached end of the stream
                frames.append(buffer)

            # read audio input until the phrase ends
            pause_count, phrase_count = 0, 0
            phrase_start_time = elapsed_time
            while True:
                # handle phrase being too long by cutting off the audio
                elapsed_time += seconds_per_buffer
                if phrase_time_limit and elapsed_time - phrase_start_time > phrase_time_limit:
                    break

                buffer = source.stream.read(source.CHUNK)
                if len(buffer) == 0: break  # reached end of the stream # contando o buffer pela segunda vez 
                frames.append(buffer)
                phrase_count += 1

                # check if speaking has stopped for longer than the pause threshold on the audio input
                energy = audioop.rms(buffer, source.SAMPLE_WIDTH)  # unit energy of the audio signal within the buffer
                if energy > self.energy_threshold:
                    pause_count = 0
                else:
                    pause_count += 1
                if pause_count > pause_buffer_count:  # end of the phrase
                    break

            # check how long the detected phrase is, and retry listening if the phrase is too short
            phrase_count -= pause_count  # exclude the buffers for the pause before the phrase
            if phrase_count >= phrase_buffer_count or len(buffer) == 0: break  # phrase is long enough or we've reached the end of the stream, so stop listening


        # obtain frame data
        for i in range(pause_count - non_speaking_buffer_count): frames.pop()  # remove extra non-speaking frames at the end

        frame_data = b"".join(frames) #bytessss
        frame_data_ds = self.delay_sum_best_guess(frame_data,
                                                  angle_range,
                                                  nChannels=source.nChannels,
                                                  sample_rate=source.SAMPLE_RATE,
                                                  sample_width=source.SAMPLE_WIDTH) 
    
        return AudioData(frame_data_ds, source.SAMPLE_RATE, source.SAMPLE_WIDTH)


    


### https://gist.github.com/hadware/8882b980907901426266cb07bfbfcd20#file-bytes_to_wav-py

