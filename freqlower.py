# In-progress code: Yue Jiang, Patricia Rhymer, Mel Chua
# Implementing various frequency lowering techniques in Python
# So far:
# low-pass filter: done
# linear compression: done
# nonlinear compression: TODO
# transposition: TODO

from numpy import *
import scipy
from scipy.io.wavfile import read, write
import matplotlib.pyplot as pyplot
from pylab import specgram

## SETUP ##

# file to test with
infile = 'baby.wav'
samplerate = 32000 # Default

# Keeping track of variables
# orig_time -- original signal, time domain
# orig_freq -- original signal, frequency domain
#           typically orig_time run through fft
# orig_time_regen -- original signal regenerated
#           to test that it reconstructs correctly
#           typically orig_freq run through ifft

## FUNCTION DEFINITIONS ##

def getwavdata(file):
    return scipy.io.wavfile.read(file)[1]

def makewav(data, outfile, samplerate):
    scipy.io.wavfile.write(outfile, samplerate, data)

def makegraph(data, filename):
    pyplot.clf()
    pyplot.plot(data)
    pyplot.savefig(filename)

def makegraphdb(data, filename):
    makegraph(10*log10(abs(data)), filename)

def makespec(data, filename):
    # Note that this function takes in the signal in the time domain
    pyplot.clf()
    sgram = specgram(data)
    pyplot.savefig(filename)

## CODE ##

# Read in original data
orig_time = getwavdata(infile)
# Convert to frequency domain
orig_freq = fft.rfft(orig_time)
# Plot in frequency domain
makegraphdb(orig_freq, 'orig_freq.png')
# Plot spectrogram -- note that spectrograms take in the
# time domain signal
makespec(orig_time, 'orig_spec.png')

# ----------------------------------------------------------------------
# Step 1: Make sure we can take a signal from the time domain to the
# frequency domain and back -- and that it'll sound the same when
# it gets back to the time domain.
# ----------------------------------------------------------------------

# Convert back to time domain
orig_time_regen = array(fft.irfft(orig_freq, len(orig_time)),dtype=int16)

# Output .wav file to verify that we reconstruct the signal correctly
# these two .wav files should sound identical to the original .wav file
makewav(orig_time, 'orig_time.wav', samplerate)
makewav(orig_time_regen, 'orig_time_regen.wav', samplerate)

# ----------------------------------------------------------------------
# Step 2: Low-pass filter
# ----------------------------------------------------------------------

# Make a copy of the original signal in the frequency domain
lowpass_freq = orig_freq

# Remove everything past the 10000th index (in the frequency domain)
lowpass_freq[10000:] = 1 # using 1 instead of 0 to keep graph length the same
                         # if we used 0's the graph would truncate at the
                         # cutoff frequency
# Note that we have no idea what frequency this corresponds to -- we
# picked 10000 because the frequency domain graph (orig_freq.png)
# and spectrogram (orig_spec.png) seemed to have most of the speech
# information underneath the 10000th frequency domain array element

# Convert back to time domain
lowpass_time = array(fft.irfft(lowpass_freq, len(orig_time)),dtype=int16)

# Output wav of low-pass filtered signal
makewav(lowpass_time, 'lowpass_time.wav', samplerate)
# Output frequency spectrum of low-pass filtered signal
makegraphdb(lowpass_freq, 'lowpass_freq.png')
# Output spectrogram of low-pass filtered signal
makespec(lowpass_time, 'lowpass_spec.png')


# ----------------------------------------------------------------------
# Step 3: Compression
# ----------------------------------------------------------------------

# Through our low-pass filter, we found most of the speech information
# takes place below the 10000th index of the signal in the frequency domain.
# Take those first 10000 datapoints for the frequency domain and compress
# them by a factor of 2. This gives us an array of length 10000/2 = 5000.
comp_freq = orig_freq[0:10000:2]
# Pad the remainder of the array with ones so it's the same length
# as the original.
paddinglength = len(orig_freq) - len(comp_freq)
comp_freq = concatenate([comp_freq, ones(paddinglength, dtype=int16)])

# Convert back to time domain
comp_time = array(fft.irfft(comp_freq, len(orig_time)),dtype=int16)

# Output wav of compressed signal
makewav(comp_time, 'comp_time.wav', samplerate)
# Output frequency spectrum of compressed signal
makegraphdb(comp_freq, 'comp_freq.png')
# Output spectrogram of compressed signal
makespec(comp_time, 'comp_spec.png')
