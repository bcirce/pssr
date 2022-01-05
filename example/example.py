#%%
from pssr import pssr 
from speech_recognition import UnknownValueError, RequestError, Recognizer

print('oi')
r = Recognizer() #recognizes audio, outputs transcript
ps = pssr.PSRecognizer() #PSRecognizer instance to listen and generate the audio
psmic = pssr.PSMic(nChannels=3) #ps eye mic array

with psmic as source:
    print('*recording')
    audio = ps.listen(source) 
    print('*done recording')

try:
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    print("Google Speech Recognition thinks you said ") 
    print(r.recognize_google(audio, language='de-DE',show_all=True))
except UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))


