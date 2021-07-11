# Machine learning audio compression
A novel approach to audio compression in bandwidth constrained circumstances through replacing conventional data streams with machine learning training and inference at both ends of communication.

# Overview

Conventional audio codecs, particularly those meant for low bandwidth consumption, encode and lossy compress the stream of samples from a recording device so that it can be reconstructed as a stream of samples on the receivers end.

Our solution aims to replace this stream through a machine learning model.
Essentially, When a VoIP call between 2 individuals starts, a model is trained in real time as they begin to talk. This model aims to create a clone of their voice that can be used to synthesize any syllable or string of syllables together so that it sounds like words and sentences spoken in the senders voice.

# Literature

## [Mozilla TTS](https://github.com/mozilla/TTS)

One of the biggest open source projects for text to speech synthesis. Then can be used as a reference for synthesizing our voice cloning.

## [Nvidia AI video compression](https://developer.nvidia.com/ai-video-compression)

An inspiration for this project. Essentially what Nvidia did was stead of sending a video stream as a sequence of frames, to send a single image of the sender's face followed with a stream of how their face moves.
A machine learning model on the receivers end puppeteers the sent image according to how their face moves.

## [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf)

## [An implementation of WaveNet](https://github.com/NVIDIA/tacotron2)

## [Realtime voice cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)

## [Opus codec](https://opus-codec.org/)

This is the most commonly used codec in most modern applications. This can be considered the benchmark to compare our approach to.

# Contribution
As far as we can tell there is no direct project that has done a directly comparable concept.
We believe this application can prove useful to serve more reliable and effective means of VoIP in cases of slow and bad connectivity.

Although you could find open source modules that offer functionality that contributes to this approach, they are not catered to this application and would not give satisfactory performance.
For example, voice cloning models do not particularly give attention to the size of the voice descriptor, something that is critical to this application.
Furthermore, Our approach requires faster processing for training the model as well as narrating the syllables back.

Some of the practical examples of this application include how the auto is interpreted through phonetic symbols. Each of these symbols take some time to play back. This gives more time for the next syllable to be received. In other words, in case of large degrees of jitter and packet loss, you would still hear complete syllables unlike how in conventional codecs, you would get a few stuttering sounds that would not necessarily resemble speech. This means this approach produces more intelligible in those extreme cases.