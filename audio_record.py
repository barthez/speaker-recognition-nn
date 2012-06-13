#!/usr/bin/python2



import sys
import time
import getopt
import alsaaudio
from cStringIO import StringIO
from os import tmpfile
from numpy import fromstring, int16
import scipy.io.wavfile as wf


def record(card = 'default', chl = 1, sampling = 44100):
    """ 
    Record one minute of sound or until KeyboardInterrupt (Ctrl+C). Return
    sampling frequency and sound as numpy.array(dtype=int16)

    Return tuple: (fs, data)
    """
    f = StringIO()
    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK, card)

    inp.setchannels(chl)
    inp.setrate(sampling)
    inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)

    inp.setperiodsize(160)
    
    try:
        it = sampling*60; #one minute
        while it>0:
            # Read data from device
            l, data = inp.read()
            it -= l
            if l:
                f.write(data)
                time.sleep(.001)
    except KeyboardInterrupt:
        pass

    return (sampling, fromstring(f.getvalue(), dtype=int16))

if __name__ == '__main__':
    print "Stop recording by pressing Ctrl+C"
    (fs, snd) = record()
    fn = raw_input("Enter filename: ")
    wf.write(fn, fs, snd)
    print "File saved correctly"

    
    
    pass
