# PyCon 2013 -- Mel Chua's Signal Processing workshop

# let's import the libraries we need to do fourier transforms
from numpy import *

# let's create some data -- here we're making a signal
# that is 2 added tones (1250Hz and 625Hz, at a 10kHz sample rate)
x = arange(256.0)
sig = sin(2*pi*(1250.0/10000.0)*x) + sin(2*pi*(625.0/10000.0)*x)

# what does this look like?
# let's import the plotting libraries
import matplotlib.pyplot as pyplot

# and then plot the signal...
pyplot.plot(sig)
pyplot.savefig('sig.png')

# and now we're done, so let's clear the plot.
pyplot.clf()

# while we're at it, let's make a graphing function
def makegraph(data, filename):
    pyplot.clf()
    pyplot.plot(data)
    pyplot.savefig(filename)

# let's take the FFT of the signal
# notice it's a real-valued input, so we do RFFT
data = fft.rfft(sig)
makegraph(data, 'fft0.png')

# note that we had 256 datapoints in sig (because x = arange(256.0))
# but we're only graphing 127. That's because in real-valued
# functions, the output is symmetric -- rfft only computes one,
# not the duplicates.

# we're interested in the magnitude, so...
data = abs(data)
makegraph(data, 'fft1.png')

# whoa, what happened? we went from a jaggedy line to two peaks...
# ...and the value of the peaks is way higher? that first peak
# went from close to -3 to close to 130! what gives?
# also, why are the peaks suddenly the same size?

# well, the fft gives us a complex number.
type(fft.rfft(sig)[0])

# and abs takes the magnitude of that complex number.
# it returns real numbers.
type(abs(fft.rfft(sig)[0]))

# if we take a look, we can see it's the 16th and 32nd values
# that are the peaks in both graphs...
fft.rfft(sig)
abs(fft.rfft(sig))

# so let's look at them both here.
# this is an imaginary number, and only the real part is plotted
# but it's got a large imaginary part
fft.rfft(sig)[16]

# so when we take the magnitude with abs, that imaginary part adds in.
abs(fft.rfft(sig)[16])

# and when we look at the 32nd value...
fft.rfft(sig)[32]
abs(fft.rfft(sig)[32])

# now, back to our graph. we see two peaks. Good!
# So, we're seeing the two peaks... but this is raw power output
# Let's normalize to decibels -- that's how we usually think of volume
# http://en.wikipedia.org/wiki/Decibel
# decibels are 10*log10, so...
data = 10*log10(data)
makegraph(data, 'fft2.png')

# hey look, quantization noise! We'll talk about this in a little bit...

# recall that we added 2 sinusoids at 625 and 1250 Hz...
# do those peaks correspond to those sinusoids on the graph?

# well, the numpy fft function goes from 0-5000Hz
# so the X-axis is marked in increments that correspond to
# values of 0-5000Hz divided into 128 slices
# 5000/128 = 39.0625 Hz per tick mark
# the two peaks here are:
# (5000/128)*16 = 625 Hz
# (5000/128)*32 = 1250 Hz
# which are the tones we used, so we win!

# this smushes all the times together, what if we want to see them again?
# here's another visualization called a spectrogram...
# STOP: SLIDE
from pylab import specgram
pyplot.clf()
sgram = specgram(sig)
pyplot.savefig('sgram.png')

# do you see how the spectrogram becomes the spectrum when sliced?

# now let's do this with a more complex sound.
# https://www.pythonanywhere.com/user/mchua/files/home/mchua/flute.wav

# we need libraries for working with wave files
# http://docs.scipy.org/doc/scipy/reference/io.html
import scipy

# later on, I will show you how to use the wave library
# and why you don't want to (during break)

# how to import it...
def getwavdata(file):
    return scipy.io.wavfile.read(file)[1]

# let's take a closer look at what's going on
# data = scipy.io.wavfile.read('flute.wav') returns a tuple
# the first number is the sampling rate (samples/second)
# notice it's 44100, which is CD-quality samping
# the second is a numpy array with the data (dtype int16)
# (bonus: why int16? see http://docs.scipy.org/doc/numpy/user/basics.types.html)
# and think about space-saving.

# So let's get the data...
audio = getwavdata('flute.wav')

# hang on!
# how do we make sure we've got the same data out as in?
# well, we could write it back to a wav file and  play it back.
# how do we do that?

def makewav(data, outfile, samplerate):
    scipy.io.wavfile.write(outfile, samplerate, data)
  # note: this assumes the data is in a numpy array
	# if not, convert before passing to makewav
	# example: array(data, dtype=int16)
	
# let's listen
makewav(audio, 'foo.wav', 44100)

# let's see what this looks like in the time domain
makegraph(audio[0:1024], 'flute.png')

# These do sort of the same thing, btw
# I don't really care about the difference
numpy.fft.rfft(audio)
scipy.fftpack.rfft(audio)

# What does the fft look like?
audiofft = fft.rfft(audio)
makegraph(audiofft, 'flute_fft.png')

# oh wait, we wanted to normalize this
audiofft = abs(audiofft)
audiofft = 10*log10(audiofft)
makegraph(audiofft, 'flute_fft.png')

# and the spectrogram?
pyplot.clf()
sgram = specgram(audio)
pyplot.savefig('flutespectro.png')

# what note is the flute playing?
# http://www.bgfl.org/custom/resources_ftp/client_ftp/ks2/music/piano/flute.htm
# sounds like a B
# which is about 494Hz
# http://en.wikipedia.org/wiki/Piano_key_frequencies

# MID-WORKSHOP BREAK

# NOTES PEOPLE ARE FINDING:

# sin generates really, really tiny amplitudes
# (compare to the magnitude of the data you get when
# you read in the flute wavfile)
# how can you increase the amplitude of your signal?

# also: once you get an audible .wav file, does it sound like
# a clean tone? (What do you think might be the cause?)

def makesinwav(freq, amplitude, sampling_freq, num_samples):
    return array(sin(2*pi*freq/float(sampling_freq)*arange(float(num_samples)))*amplitude,dtype=int16)
    

# if you're having trouble getting your sine wave to sound like a pure tone...
def savewav(data, outfile, samplerate):
    #coerce to dtype int16 from float value
    out_data = array(data, dtype=int16)
    scipy.io.wavfile.write(outfile, samplerate, out_data)

# sample rate illustrations
audio = getwavdata('flute.wav')
makewav(audio, 'fluteagain44100.wav', 44100)
makewav(audio, 'fluteagain22000.wav', 22000)
makewav(audio, 'fluteagain88200.wav', 88200)

# Refresher: the fft takes us from the time
# to the frequency domain
flutefft = fft.rfft(audio)

# Going... backwards!
# Using the inverse fast fourier transform to go
# from the frequency to the time domain
reflute= fft.irfft(flutefft, len(audio))
reflute_coerced = array(reflute, dtype=int16) # coerce it
makewav(reflute_coerced, 'fluteregenerated.wav', 44100)

# We can now look at the graph of our flute sound
# in the frequency domain...
makegraph(10*log10(abs(flutefft)), 'flutefftdb.png')

# ...decide to low-pass filter it...
flutefft[5000:] = 0

# and see what the frequency spectrum looks like now.
makegraph(10*log10(abs(flutefft)), 'flutefft_lowpassed.png')

# go back to the time domain so we can listen
# to the low-passed flute sound
reflute= fft.irfft(flutefft, len(audio))
reflute_coerced = array(reflute, dtype=int16) # coerce it
makewav(reflute_coerced, 'flute_lowpassed.wav', 44100)

# TODO: VOCODER CODE - "AVE MARIA"

---

# filtering noisy things

import random
x = arange(256.0)
sig = sin(2*pi*(1250.0/10000.0)*x) + sin(2*pi*(625.0/10000.0)*x)
noisy = [x/2.0 + random.random()*0.1 for x in sig]

pyplot.clf()
sgram = specgram(sig)
pyplot.savefig('nonoise.png')

pyplot.clf()
sgram = specgram(noisy)
pyplot.savefig('withnoise.png')

# looks awful!
# okay, let's go back to something simpler

def lowpassfilter(signal):
   for i in range(20, len(signal)-20):
      signal[i] = 0


######### END OF TUTORIAL ##########
######### BEGIN UNUSED CODE SNIPPETS ########
# (created just in case we needed them)

## Appendix 1: A more computationally efficient way to
## generate a sine wave

# a more efficient way to make a signal
duration = 4 # seconds
samplerate = 44100 # Hz
samples = duration*samplerate
frequency = 440 # Hz

# just compute one period
period = samplerate / float(frequency) # in sample points
omega = 2*pi / period # 2*pi*frequency / samplerate
xaxis = N.arange(int(period),dtype = N.float) * omega
ydata = 16384 * N.sin(xaxis)

# and then repeat it
signal = N.resize(ydata, (samples,))
	
## Appendix 2: Hanning windows

# http://en.wikipedia.org/wiki/Window_function#Hann_.28Hanning.29_window

from scipy.signal import hann
audio = getwavdata('flute.wav')
window = hann(len(audio))
audio = audio * window

## Appendix 3: Labeling graphs

# If you want to have labels for your graphs...

# the graph
pyplot.plot(audio)
# label the axes
pyplot.ylabel("Amplitude")
pyplot.xlabel("Time (samples)")
# set the title
pyplot.title("Flute Sample")

## Appendix 4: useful terms to look up

# We didn't have time to discuss these, but if you
# do further work in signal processing, you should
# know these vocabulary words

# impulse response
# transfer function
# envelope
