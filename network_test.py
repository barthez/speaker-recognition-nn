#!/usr/bin/python2

from ffnet import ffnet, mlgraph, savenet, loadnet
from numpy import zeros, ones, argsort
import voice_sample
import re

LPC_COE_NUM = 12

def get_inputs_and_outputs(samples, word, range_start, range_end):
    sn = len(samples) #samples (person) number
    rw = (range_end - range_start + 1) #Range width
    row_num = sn*rw
    outputs = zeros( (row_num, sn) )
    inputs = zeros( (row_num, LPC_COE_NUM) )
    for n, p in zip(range(sn), samples):
        a = n*rw
        b = (n+1)*rw
        outputs[a:b, n] = ones(rw)
        inputs[a:b, :] =p.get_network_inputs(word, LPC_COE_NUM,
                range(range_start-1, range_end) )

    return (inputs, outputs)



def train_net_a_person(net, person, pn, word, rg):
    output = zeros(len(net.outno))
    output[pn] = 1.0
    outputs = [ output for i in rg ]
    inputs = person.get_network_inputs(word, LPC_COE_NUM, rg) 
    net.train_momentum(inputs, outputs)
    return net

def learn_net(filename, word):
    people = voice_sample.get_names(); 
    people_num = len(people)
    try:
        network = loadnet(filename)
    except IOError, ex:
        print "Nie ma takiego pliku, tworzenie nowej sieci"
        network = ffnet(mlgraph((LPC_COE_NUM, people_num + LPC_COE_NUM,
            people_num)) )

    print "Loading voice samples..."
    samples = [ voice_sample.VoicePersonSamples(s) for s in people]
    ans = raw_input("Select sample range (1-10):  ")
    print ans
    (rs, re) = map(lambda x: int(x), ans.strip().split("-"))

    (inp, out) = get_inputs_and_outputs(samples, word, rs, re);

    network.train_momentum(inp, out)

    savenet(network, filename)

def test_net(filename, word):
    people = voice_sample.get_names(); 
    people_num = len(people)
    try:
        network = loadnet(filename)
    except IOError, ex:
        print "Nie ma takiego pliku, stworz wczesniej siec"
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


def main(args):
    if len(args) < 4:
        print "Usage: {:s} type filename word"
        print "\ttype: learn test"
        print "\tfilename: network filename"
        print "\tword: word for learnig/testing (imie, nazwisko, samochod, kot)"
        return 1
    cmd = args[1]
    filename = args[2]
    word = args[3]
    if cmd == 'learn':
        learn_net(filename, word)
    elif cmd == 'test':
        test_net(filename, word)


if __name__ == "__main__":
    import sys
    main(sys.argv)
