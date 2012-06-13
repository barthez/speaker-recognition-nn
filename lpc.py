import pylab, scipy, spectrum
import voice_sample

def frame(x, framesize, framehop):
    w = scipy.hamming(framesize)
    return scipy.array([w*x[i:i+framesize] 
                     for i in range(0, len(x)-framesize, framehop)])

if __name__ == '__main__':
    import scipy.io.wavfile as wf
    names = voice_sample.get_names()
    for num, name in zip(range(1, len(names) + 1), names):
        pylab.subplot(2,3, num)
        for it in range(0, 9):
            (fs, sd) = wf.read(voice_sample.SAMPLES_DIR + '/%s/imie/%02d.wav' % (name, it+1) )
            sd = sd - scipy.mean(sd)
            sd = sd / scipy.amax(sd)
            #sf = frame(sd, int(fs*0.03), int(fs*0.02))

            #lpcc = scipy.array([ lpc(f, 10) for f in sf ]);
            #lpcc_mean = scipy.mean(lpcc, 0)

            #pylab.figure();
            #pylab.plot(sd);
            lpcc, e = spectrum.lpc(sd, 12)
            pylab.plot(lpcc, label="Zapis {:02d}".format(it+1) );
        pylab.title(name)
        pylab.ylim(-5, 5)

    pylab.show();

