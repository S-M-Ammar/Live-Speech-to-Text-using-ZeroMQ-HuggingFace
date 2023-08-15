import asyncio
import zmq
import zmq.asyncio
import os 
import torch
import torchaudio
import pandas as pd
import numpy as np
from transformers import pipeline , AutoProcessor
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

pipe = pipeline("automatic-speech-recognition", model="codenamewei/speech-to-text")

ctx = zmq.asyncio.Context()
socket = ctx.socket(zmq.PULL)
socket.bind("tcp://*:5555")


SAMPLING_RATE = 16000

# def process_audio(audio_array):
#     processed_audio = processor(audio_array, sampling_rate=SAMPLING_RATE).input_values[0]
#     return processed_audio

async def recv_and_process():
    print("Voice transcriber is up and running .... \n")
    while True:
        frames = await socket.recv_json() # waits for msg to be ready
        # audio_bytes = await socket.recv() # for bytes
        try:
            print(f"Message Recieved")
            text = ''
            frames = np.array(frames['frames'])
            frames = frames.astype(float)
            for frame in frames:
                text = text + pipe(frame)['text']
            print(text)
        except Exception as e:
            print(f"Error : {e}")



loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    loop.run_until_complete(recv_and_process())
finally:
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()