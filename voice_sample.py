#!/usr/bin/python2 
# coding=utf8
#

from scipy.io import wavfile
from spectrum import lpc
from numpy import array
import numpy as np
from os import listdir
from os.path import isdir
from os.path import join as path_join
from re import match
import audio_record
from spectrogram import stft
from datetime import datetime

import pylab

WORDS = ['imie', 'nazwisko', 'samochod', 'kot']
SAMPLES_DIR = './samples'

def timestamp(fmt="%Y%m%d%H%M%S"):
    return datetime.now().strftime(fmt)

def get_names():
    return [person for person in listdir(SAMPLES_DIR)]



def find_words(data, fs, num, **args):
    min_silence_length = args["min_silence_length"] if "min_silence_length" in args else 5
    _plot = args["plot"] if "plot" in args else False
    _debug = False

    frml = 0.03
    frmh = 0.02

    X = stft(data, fs, frml, frmh)
    magnitude = np.absolute(X)
    signal_energy = np.sum(magnitude, 1)


    if "thresh" in args.keys():
        thresh = args["thresh"]
    else:
        thresh = np.mean(signal_energy) - np.min(signal_energy)

    en_th = (signal_energy > thresh).astype(np.int8)
    rising_edge = np.diff( np.hstack(([0], en_th)) )
    falling_edge = np.diff( np.hstack((en_th, [0])) )

    (stp,) = np.nonzero(rising_edge > 0)
    (edp,) = np.nonzero(falling_edge < 0)

    if len(stp) != len(edp):
        raise ValueError("Wrong threshold value")

    """Remove too short silence periods"""
    silence = stp[1:] - edp[:-1]
    (too_short,) = np.nonzero(silence <= min_silence_length)
    stp = np.delete(stp, too_short + 1)
    edp = np.delete(edp, too_short)

    nonsilence = edp - stp
    shortest = nonsilence.argsort()
    
    if len(nonsilence) < num:
        raise ValueError("You say less words than you ask me to find or you speak too fast")
    """ Remove too short nonsilence """
    stp = np.delete(stp, shortest[:-num])
    edp = np.delete(edp, shortest[:-num])

    pos = lambda a: fs*frmh*a
    
    if _plot:
        pylab.subplot(2,1,1)
        etime = np.array(range(len(signal_energy)), dtype=float)*frmh
        pylab.plot(etime,signal_energy)
        pylab.axhline(thresh)
        [ pylab.axvline(x*frmh, color="red") for x in stp ]
        [ pylab.axvline(x*frmh, color="green") for x in edp ]
        pylab.xlabel("Czas [s]")
        pylab.title("Energia wypowiedzi w czasie")
        time = np.array(range(len(data)), dtype=float)/fs
        sc_data = data - np.mean(data)
        sc_data = sc_data/np.max(sc_data)
        pylab.subplot(2,1,2)
        pylab.plot(time, sc_data)
        [ pylab.axvline(pos(x)/fs, color="red") for x in stp ]
        [ pylab.axvline(pos(x)/fs, color="green") for x in edp ]
        pylab.xlabel("Czas [s]")
        pylab.title(u"Zapis sygnału")
        pylab.show()


    return [ data[pos(a):pos(b)] for a,b in zip(stp,edp) ]


class VoiceSample:
    def __init__(self, _filename, _typename, **args):
        if not _typename in WORDS:
            raise ValueError('Type is not one of premitted types: {:s}'
                    .format(", ".join(WORDS)))
        self.typename = _typename
        self.filename = _filename
        argn = args.keys();
        if "data" in argn and "fs" in argn:
            self.fs = args["fs"]
            self.data = args["data"]
        else:
            (self.fs, self.data) = wavfile.read(_filename);

    def __len__(self):
        return len(self.data)

    def lpc_coeff(self, p):
        print "LPC coeff for: " + str(self)
        sc_data = self.data - np.mean(self.data)
        sc_data = sc_data/np.max(sc_data)
        (coeff, err) = lpc(sc_data, p);
        return coeff

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<VoiceSample: filename: '{:s}', sampling freq {:d} Hz, length: {:d} ms>".format(
                    self.filename,
                    self.fs, 
                    len(self)*1000/self.fs)
    def save(self):
        wavfile.write(self.filename, self.fs, self.data)

    @classmethod
    def record_sample(cls, _typenames):
        (fs, sound) = audio_record.record()
        tns = _typenames if type(_typenames) == list else [_typenames]
        word_num = len(tns)
        
        create_voice_sample = lambda n,t,f,w: VoiceSample(
                "recording_{name}_{num:02d}_{tmstp}.wav".format(name=t,
                    num=n, tmstp=timestamp()),
                t,
                fs=f,
                data=w)
        collection = zip(
                range(1,word_num+1),
                tns,
                find_words(sound, fs, word_num, plot=True))

        return [ create_voice_sample(n,t,fs,w) for n,t,w in collection ]

        



class VoiceTypeSamples(tuple):
    def __init__(self, _name, _type):
        tuple.__init__(self)
        self.name = _name
        self.typename = _type

    def __new__(cls, _name, _type):
        smpl = cls.__load_samples(_name, _type)
        return tuple.__new__(cls, smpl)

    @classmethod
    def __load_samples(cls, n, t):
        samples_path = path_join(SAMPLES_DIR, n, t)
        samples = []
        paths = [path_join(samples_path, fname) for fname in
                listdir(samples_path)]
        paths.sort()
        for fname in paths:
            if match("^.*\.wav$", fname):
                samples.append( VoiceSample( fname, t ) )
        return samples

class VoicePersonSamples(dict):
    def __init__(self, _name):
        dict.__init__(self)
        self.name = _name;
        self.real_name = " ".join( map(lambda x: x[0].upper() + x[1:],
            _name.split('_') ) )
        samples_path = path_join(SAMPLES_DIR, self.name)
        for fname in listdir(samples_path):
            if isdir(path_join(samples_path, fname)):
                    self[fname] = VoiceTypeSamples(self.name, fname)

    def get_network_inputs(self, word, size, rg):
        if not (word in WORDS):
            raise ValueError('Type is not one of premitted types: {:s}'
                    .format(", ".join(WORDS)))
        #print "Get inputs for word: {:s}".format(word)
        return array([ self[word][i].lpc_coeff(size) for i in rg])

def __main(args):
    print "Program testowy"
    #bartek = VoiceSample('./bartek_bulat/imie/01.wav', 'imie')
    #bartek = VoicePersonSamples('bartek_bulat')
    #print bartek.real_name

    #(fs, data) = wavfile.read("pl3.wav")
    #words = find_words(data, fs, 1)
    #wavfile.wrie("cleared_pl3.wav", fs, words[0])
    raw_input("To start recording press [Enter], to stop [Ctrl+C]")
    print "Recording"
    words = VoiceSample.record_sample(["imie", "samochod"])
    map(lambda w: w.save(), words)
    



if __name__ == "__main__":
    import sys
    __main(sys.argv)
