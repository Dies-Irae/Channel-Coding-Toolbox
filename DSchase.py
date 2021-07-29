import numpy as np
import QRCode
import Simulation
import Chase


def test_thread(code_len, batch_size, SNR_start, SNR_stop, step, max_errs):
    """
    Single test thread for DS algorithm
    :param code_len: Code Length
    :param batch_size: Batch size for every loop
    :param SNR_start: the start point snr, start is included
    :param SNR_stop: the end point snr, end is included
    :param step: step size
    :param max_errs: if max errs achieved, end
    :return: BER Array(Numpy array)
    """
    np.random.seed()
    qr = QRCode.QRCode(code_len)
    simul = Simulation.Simulation(qr.n, qr.k, qr.G, qr.H)
    chase = Chase.Chase(qr, qr._DSAlgorithm)
    SNRs = np.arange(SNR_start, SNR_stop + step, step)
    resErr = np.zeros(SNRs.shape[0])
    resBlk = np.zeros(SNRs.shape[0])
    index = 0
    for i in SNRs:
        errs = 0
        blks = 0
        while errs < max_errs:
            sample = simul.generateBatch(batch_size)
            encodedSample = simul.encode(sample)
            received = simul.AWGN(encodedSample, i)
            decoded = np.zeros_like(received)
            for j in range(received.shape[0]):
                decoded[j] = chase.decode(received[j])
            BER = np.mean(np.logical_xor(encodedSample, decoded))
            errs += BER * code_len * batch_size
            blks += batch_size
        resErr[index] = errs
        resBlk[index] = blks
        index += 1
    return resErr, resBlk


##TEST###
if __name__ == "__main__":
    import multiprocessing as mp
    import time

    print(time.asctime(time.localtime(time.time())))
    num_cores = int(mp.cpu_count())
    print("Total Cores: " + str(num_cores) + " Cores")
    n_workers = 5
    code_len = 47
    batchSize = 100
    maxErrs = (250 // n_workers) + 1
    SNR_start = 1
    SNR_stop = 1
    step = 1

    SNRs = np.arange(SNR_start, SNR_stop + step, step)
    resErr = np.zeros(SNRs.shape[0])
    resBlk = np.zeros(SNRs.shape[0])
    res = np.zeros(SNRs.shape[0])
    pool = mp.Pool(n_workers)
    results = []
    for _ in range(n_workers):
        results.append(pool.apply_async(test_thread, args=(code_len, batchSize, SNR_start, SNR_stop, step, maxErrs)))
    for worker in results:
        errs, blks = worker.get()
        resErr += errs
        resBlk += blks
    for i in range(SNRs.shape[0]):
        print("BER: %e @ %f dB" % (resErr[i] / resBlk[i] / code_len, SNRs[i]))
    print(time.asctime(time.localtime(time.time())))
