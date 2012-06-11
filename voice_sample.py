#!/usr/bin/python2 

from scipy.io import wavfile
from spectrum import lpc
from numpy import array
from os import listdir
from os.path import isdir
from os.path import join as path_join
from re import match

SAMPLES_DIR = './samples'

def get_names():
    return [person for person in listdir(SAMPLES_DIR)]

class VoiceSample:
    def __init__(self, _filename, _typename):
        self.typename = _typename
        self.filename = _filename
        (self.fs, self.data) = wavfile.read(_filename);

    def __len__(self):
        return len(self.data)

    def lpc_coeff(self, p):
        (coeff, err) = lpc(self.data, p);
        return coeff

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<VoiceSample: filename: '{:s}', sampling freq {:d} Hz, length: {:d} ms>".format(
                    self.filename,
                    self.fs, 
                    len(self)*1000/self.fs)

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
        for fname in listdir(samples_path):
            if match("^.*\.wav$", fname):
                samples.append( VoiceSample( path_join(samples_path, fname), t ) )
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
        return array([ self[word][i].lpc_coeff(size) for i in rg])

def __main(args):
    print "Program testowy"
    #bartek = VoiceSample('./bartek_bulat/imie/01.wav', 'imie')
    bartek = VoicePersonSamples('bartek_bulat')
    print bartek.real_name

if __name__ == "__main__":
    import sys
    __main(sys.argv)
