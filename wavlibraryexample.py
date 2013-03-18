# Using http://docs.python.org/2/library/wave.html
# aka "why we use scipy"
# an alternative implementation of the functions getwavdata and makewav
# as defined in sigproc-outline.py

import wave
import struct

# performs the same function as getwavdata
# using the wave library instead of scipy
def wgetwavdata(file):
    # open a file for writing
  infile = wave.open(file, 'r')
	length = infile.getnframes()
	data = []
	for i in range(length):
	    nextframe = infile.readframes(1)
		# this returns C-style little-endian shorts, so we convert...
		data += struct.unpack("<h", nextframe)
	return data

# performs ALMOST the same function as makewav
# using the wave library instead of scipy
# note that the 3rd argument is params rather than samplerate
# and that the 3rd element of params is the sample rate
# (see notes below)
def wmakewav(data, outfile, params=(1,2,44100, 0, 'NONE', 'not compressed')):
    wavfile = wave.open(outfile, 'w')
	wavfile.setparams(params)
	# convert back to C-style little-endian shorts...
	converted = [struct.pack("<h", i) for i in data]
	outdata = ''
	for i in converted:
	    outdata += i[0]
		outdata += i[1]
	wavfile.writeframes(outdata)
	wavfile.close()

# Regarding params:
# arguments are (nchannels, sampwidth, framerate, nframes, comptype, compname)
# nchannels is 1 for stereo and 2 for mono
# sampwidth is the number of bytes per sample, usually 1 or 2
# framerate is the sample rate, 44100 is common for CDs
# nframes is the number of frames
# comptype is compression type
# compname is the human-readable name for comptype
