#!/usr/bin/python2

from ffnet import ffnet, mlgraph, savenet, loadnet
from numpy import zeros, ones, argsort, array, size
from numpy.random import shuffle
import voice_sample
import re
from getopt import getopt
LPC_COE_NUM = 13

def load_speaker_recognition_newtork(filename, create_new=False):
    """
    Load or create (if you  want) network for speker recognition form file

    returns tuple: (network, people_names, people_number)
    """
    people = voice_sample.get_names(); 
    people_num = len(people)
    network = None

    try:
        network = loadnet(filename)
    except IOError, ex:
        if create_new:
            network = ffnet(mlgraph((LPC_COE_NUM, people_num + LPC_COE_NUM,
                people_num)) )
    return (network, people, people_num)

def get_inputs_and_outputs(samples, word, range_start, range_end, shuffled=True):
    sn = len(samples) #samples (person) number
    rw = (range_end - range_start + 1) #Range width
    row_num = sn*rw
    outputs = zeros( (row_num, sn) )
    inputs = zeros( (row_num, LPC_COE_NUM) )
    for n, p in zip(range(sn), samples):
        a = n*rw
        b = (n+1)*rw
        idx = array(range(n,sn*rw,sn))
        outputs[a:b, n] = ones(rw)
        inputs[a:b, :] =p.get_network_inputs(word, LPC_COE_NUM,
                range(range_start-1, range_end) )

    idx = array(range(row_num))

    shuffled and shuffle(idx)
    return (inputs[idx], outputs[idx])



def train_net_a_person(net, person, pn, word, rg):
    output = zeros(len(net.outno))
    output[pn] = 1.0
    outputs = [ output for i in rg ]
    inputs = person.get_network_inputs(word, LPC_COE_NUM, rg) 
    net.train_momentum(inputs, outputs)
    return net

def learn_net(filename, word):
    (network, people, people_num) = load_speaker_recognition_newtork(filename, True);

    print "Loading voice samples..."
    samples = [ voice_sample.VoicePersonSamples(s) for s in people]
    ans = raw_input("Select sample range (1-10):  ")
    print ans
    (rs, re) = map(lambda x: int(x), ans.strip().split("-"))
    
    (inp, out) = get_inputs_and_outputs(samples, word, rs, re);

    network.train_momentum(inp, out)

    savenet(network, filename)

def test_net(filename, word):
    (network, people, people_num) = load_speaker_recognition_newtork(filename)
    if not network:
        return 1

    print "Loading voice samples..."
    samples = [ voice_sample.VoicePersonSamples(s) for s in people]
    print "Which person do you want to test:"
    for p, n in zip(samples, range(people_num) ):
        print "{:d}. {:s}".format(n+1, p.real_name)

    choice = int(raw_input("Choice: "))
    s_no = int(raw_input("Select sample (1-10):  "))

    if choice > 0:
        inp = samples[choice-1][word][s_no-1].lpc_coeff(LPC_COE_NUM)
        out = network(inp)
        fit = argsort(out)[::-1]
        for arg in fit:
            print "{:s}: {:10.5f}".format( samples[arg].real_name, out[arg] )
    else:
        print "Wrong choice"

def test_and_plot():
    pass

def test_net_using_file(filename):
    (network, people, people_num) =load_speaker_recognition_newtork(filename)
    if not network:
        return 1
    print "Loading voice samples..."
    samples = [ voice_sample.VoicePersonSamples(s) for s in people]
    fn = raw_input("File path:  ")
    vs = voice_sample.VoiceSample(fn, 'imie')
    
    inp = vs.lpc_coeff(LPC_COE_NUM)
    out = network(inp)
    fit = argsort(out)[::-1]
    for arg in fit:
        print "{:s}: {:10.5f}".format( samples[arg].real_name, out[arg] )


def record_and_test(filename, word):
    (network, people, people_num) =load_speaker_recognition_newtork(filename)
    if not network:
        return 1

    print "Loading voice samples..."
    samples = [ voice_sample.VoicePersonSamples(s) for s in people]

    raw_input("To start recording press [Enter], to stop [Ctrl+C]")
    print("Recording...")
    (vs,) = voice_sample.VoiceSample.record_sample(word)
    print("")

    inp = vs.lpc_coeff(LPC_COE_NUM)
    out = network(inp)
    fit = argsort(out)[::-1]
    for arg in fit:
        print "{:s}: {:10.5f}".format( samples[arg].real_name, out[arg] )

def usage():
    print "Usage [-h] [-w word] [-a action] filename"


def main(argv):
    opts, args = getopt(argv[1:], 'hw:a:')
    action = 'learn'
    word = 'imie'
    for o, a in opts:
        if o == '-w':
            word = a
        elif o == '-a':
            action = a
        elif o == '-h':
            usage()
            return 0

    
    if not (word in voice_sample.WORDS):
        raise ValueError('Type is not one of premitted types: {:s}'
                .format(", ".join(voice_sample.WORDS)))

    if action != 'plot' and not args:
        usage()
        return 2

    filename = args[0]
    
    
    if action == 'learn':
        learn_net(filename, word)
    elif action == 'test':
        test_net(filename, word)
    elif action == 'testfile':
        test_net_using_file(filename)
    elif action == 'record':
        record_and_test(filename, word)
    elif action == 'plot':
        test_and_plot()

    return 0


if __name__ == "__main__":
    import sys
    r = main(sys.argv)
    sys.exit(r)

