from __future__ import division
import matplotlib.pyplot as plt
import numpy
from python_speech_features import sigproc
from python_speech_features import get_filterbanks,lifter,fbank
from scipy.fftpack import dct

def mfcc_without_dct(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
         nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,
         winfunc=lambda x:numpy.ones((x,))):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    feat = numpy.log(feat)
    # feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    # feat = lifter(feat,ceplifter)
    if appendEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
    return feat

def centrlized_mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
         nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True,
         winfunc=lambda x: numpy.ones((x,))):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    feat, energy = custom_fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph, winfunc)
    feat = numpy.log(feat)
    # feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    # feat = lifter(feat, ceplifter)
    if appendEnergy: feat[:, 0] = numpy.log(energy)  # replace first cepstral coefficient with log of frame energy
    return feat


def custom_fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
          nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
          winfunc=lambda x: numpy.ones((x,))):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq = highfreq or samplerate / 2
    signal = sigproc.preemphasis(signal, preemph)
    frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = sigproc.powspec(frames, nfft)
    energy = numpy.sum(pspec, 1)  # this stores the total energy in each frame
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)  # if energy is zero, we get problems with log

    # fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    fb = centerlized_mfcc(0, 4800, 8000)

    feat = numpy.dot(pspec, fb.T)  # compute the filterbank energies
    feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)  # if feat is zero, we get problems with log

    return feat, energy

def centerlized_mfcc(lowFreq=0,middleFreq=4800,highFreq=8000):
    #TODO: Make the numbers more general instead of the hardcoded that they are
    fb1 = get_filterbanks(nfilt=15,nfft=512,samplerate=16000,lowfreq=0,highfreq=middleFreq)
    fb2 = get_filterbanks(nfilt=11,nfft=512,samplerate=16000,lowfreq=0,highfreq=(highFreq-middleFreq))
    fb = numpy.zeros(257)

    for filter in fb1:
        newFilter = filter[0:150]
        newFilter = newFilter[::-1]
        zeros = numpy.zeros(len(filter)-150)
        newFilter = numpy.concatenate((newFilter,zeros),axis=0)
        fb = numpy.vstack((fb,newFilter))

    for filter in fb2:
        zeros = numpy.zeros(146)
        newFilter = filter[0:111]
        newFilter = numpy.concatenate((zeros,newFilter), axis=0)
        fb = numpy.vstack((fb, newFilter))

    return fb[1:27] #its time to give up the first element (the zeros)... I used it only for the initial structure