#!/usr/bin/python2

from spectrogram import stft
import pylab, numpy, scipy
from os import mkdir
from os.path import join as path_join
from math import floor, ceil

WORDS = ['imie', 'nazwisko', 'samochod', 'kot'];

def plot_selected(l, nst, ned, val = 4e6):
    start_plot = scipy.zeros( l )
    end_plot = scipy.zeros( l )

    for a, b, n in zip(nst, ned, range(0, len(ned))):
        if n%2 == 0:
            start_plot[a:b] = val*numpy.ones(b-a)
        else:
            end_plot[a:b] = val*numpy.ones(b-a)

    pylab.fill(start_plot, color="green")
    pylab.fill(end_plot, color="red")

def signal_energy(signal, fs, fl = 0.05, fh=0.02):
    X = stft(signal, fs, fl,fh)

    #pylab.figure();
    #pylab.imshow(scipy.absolute(X.T), origin='lower', aspect='auto',
    #                         interpolation='nearest')
    #pylab.specgram(signal, NFFT=44100*fs, noverlap=44100*fh);
    #pylab.show();

    mag = scipy.absolute(X);
    energy = scipy.sum(mag, 1)
    return energy
    

def split_into_words(filename, wn, thresh, too_short):
    import scipy.io.wavfile as wf

    (folder, ext) = filename.rsplit('.', 1);

    (fs, sig) = wf.read(filename)
    print "File is readed correctly"
    fl = 0.050
    fh = 0.020

    energy = signal_energy(sig, fs, fl, fh)
    print "Energy..."

    xx = scipy.diff(scipy.r_[ [0], (energy > thresh).astype(int) ] )
    yy = scipy.diff(scipy.r_[ (energy > thresh).astype(int), [0] ] )
    start_sig = scipy.nonzero(xx > 0)[0]
    end_sig = scipy.nonzero(yy < 0)[0]

    nst = start_sig
    ned = end_sig
    print "Enough power ranges"

   # to_less_power = [];
   # for a, b, n in zip(nst, ned, range(0, len(ned))):
   #     en =  numpy.sum(energy[a:b])
   #     print "%d: %13.2f" % (n,en/(b-a))
   #     if (en/(b-a) < 10e6):
   #         to_less_power.append(n)
   # 

   # nst = scipy.delete(nst, to_less_power)
   # ned = scipy.delete(ned, to_less_power)
   # print "Remove no enough power ranges"

    l1 = len(nst)
    l2 = len(ned)

    dist = ned- nst


    tooshort = scipy.nonzero(dist < too_short)
    nst = scipy.delete(nst, tooshort)
    ned = scipy.delete(ned, tooshort)
    
    print "Remove to shor ranges"

    dist = ned- nst
    edist = ned[1:] - ned[:-1]

    ll = len(nst) - wn
    tor = edist.argsort()[:ll]
    print tor
    print dist
    print end_sig
    print edist

    print edist[tor]

    nst = scipy.delete(nst, tor+1)
    ned = scipy.delete(ned, tor)

    print "Join ranges"

    print nst
    print ned
    print ned - nst

    words = scipy.array([sig[fs*fh*a:fs*fh*b] for a, b in zip(nst, ned)])
    
    print "Saving files"
    try:
        mkdir(folder)
        for w in WORDS:
            mkdir(path_join(folder, w ))
    except OSError, ex:
        pass

    for w, n in zip(words, range(0, len(words) ) ):
        ww = n%4
        wwig = floor(float(n)/4.0)
        flnm = path_join(folder, WORDS[ww], "%02d.wav" % (wwig+1) )
        wf.write(flnm, fs, w)
    
    print "Files saved!"
    
    pylab.figure()
    pylab.plot( energy )
    plot_selected(len(energy), nst, ned, thresh);
    pylab.show()

def main(args):
    if len(args) < 5:
        print "Usage: %s filename word_numer thresholdi too_short" % args[0]
        return 1
    filename = args[1]
    wn = int(args[2])
    thresh = float(args[3])
    too_short = int(args[4])
    split_into_words(filename, wn, thresh, too_short)

    
    


if __name__ == "__main__":
    import sys
    main(sys.argv)
