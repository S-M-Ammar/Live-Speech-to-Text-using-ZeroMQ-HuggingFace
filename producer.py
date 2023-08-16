import torch
import pandas as pd
import numpy as np
import zmq
import pyaudio

context = zmq.Context()

#  Socket to talk to server
socket = context.socket(zmq.PUSH)
socket.connect("tcp://localhost:5555")



p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16_000, input=True)
frames = None
initial_frame = True
print("Recording in progress")

while True:
    audio_chunk = stream.read(10000 , exception_on_overflow=False)  # Adjust chunk size as needed
    single_frame = np.frombuffer(audio_chunk, dtype=np.int16)
    if(initial_frame):
        initial_frame = False
        frames = single_frame
    else:
        frames = np.vstack((frames,single_frame))
    
    if(frames.shape[0]==10):
        socket.send_json({"frames":frames.tolist()})
        initial_frame = True
        frames = None
   
# print("Recording ended")

# audio_data = np.concatenate(frames)
# print(audio_data)
# print(audio_data.shape)


stream.stop_stream()
stream.close()
p.terminate()