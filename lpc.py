import pylab, scipy, spectrum

def frame(x, framesize, framehop):
    w = scipy.hamming(framesize)
    return scipy.array([w*x[i:i+framesize] 
                     for i in range(0, len(x)-framesize, framehop)])

names = ['damian_bulat', 'bartek_bulat', 'szczepan_bulat', 'aadam_traba']

if __name__ == '__main__':
    import scipy.io.wavfile as wf
    for name in names:
        pylab.figure(name)
        for it in range(0, 9):
            (fs, sd) = wf.read('%s/samochod/%02d.wav' % (name, it+1) )
            sd = sd - scipy.mean(sd)
            sd = sd / scipy.amax(sd)
            #sf = frame(sd, int(fs*0.03), int(fs*0.02))

            #lpcc = scipy.array([ lpc(f, 10) for f in sf ]);
            #lpcc_mean = scipy.mean(lpcc, 0)

            #pylab.figure();
            #pylab.plot(sd);
            lpcc, e = spectrum.lpc(sd, 13)
            pylab.plot(lpcc);

    pylab.show();

