import argparse
import time
import brainflow
import numpy as np

import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions


def main():
    BoardShim.enable_dev_board_logger()

    # use synthetic board for demo
    params = BrainFlowInputParams()
    # board_id = BoardIds.SYNTHETIC_BOARD.value
    
    params = BrainFlowInputParams()
    board_id = 38
    params.serial_number = "Muse-6D79"
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    time.sleep(10)
    data = board.get_current_board_data(DataFilter.get_nearest_power_of_two(sampling_rate))
    board.stop_stream()
    board.release_session()


    # demo how to convert it to pandas DF and plot data
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    print(eeg_channels)
    # assert 1 == 0
    df = pd.DataFrame(np.transpose(data))
    plt.figure()
    df[eeg_channels].plot(subplots=True)
    plt.savefig('before_processing.png')

    # demo for denoising, apply different methods to different channels for demo
    for count, channel in enumerate(eeg_channels):
        # first of all you can try simple moving median or moving average with different window size
        if count == 0:
            DataFilter.perform_rolling_filter(data[channel], 3, AggOperations.MEAN.value)
        elif count == 1:
            DataFilter.perform_rolling_filter(data[channel], 3, AggOperations.MEDIAN.value)
        # if methods above dont work for your signal you can try wavelet based denoising
        # feel free to try different functions and decomposition levels
        elif count == 2:
            DataFilter.perform_wavelet_denoising(data[channel], 'db6', 3)
        elif count == 3:
            DataFilter.perform_wavelet_denoising(data[channel], 'bior3.9', 3)
        elif count == 4:
            DataFilter.perform_wavelet_denoising(data[channel], 'sym7', 3)
        elif count == 5:
            # with synthetic board this one looks like the best option, but it depends on many circumstances
            DataFilter.perform_wavelet_denoising(data[channel], 'coif3', 3)

    df = pd.DataFrame(np.transpose(data))
    df.to_pickle("jack_open_denoise")
    

    for count, channel in enumerate(eeg_channels):
        # print('Original data for channel %d:' % channel)
        # print(data[channel])
        # demo for wavelet transforms
        # wavelet_coeffs format is[A(J) D(J) D(J-1) ..... D(1)] where J is decomposition level, A - app coeffs, D - detailed coeffs
        # lengths array stores lengths for each block
        wavelet_coeffs, lengths = DataFilter.perform_wavelet_transform(data[channel], 'db5', 3)
        app_coefs = wavelet_coeffs[0: lengths[0]]
        detailed_coeffs_first_block = wavelet_coeffs[lengths[0]: lengths[1]]
        # you can do smth with wavelet coeffs here, for example denoising works via thresholds 
        # for wavelets coefficients
        restored_data = DataFilter.perform_inverse_wavelet_transform((wavelet_coeffs, lengths), data[channel].shape[0],'db5', 3)
        # print('Restored data after wavelet transform for channel %d:' % channel)
        # print(restored_data)

        # demo for fft, len of data must be a power of 2
        fft_data = DataFilter.perform_fft(data[channel], WindowFunctions.NO_WINDOW.value)
        # len of fft_data is N / 2 + 1
        restored_fft_data = DataFilter.perform_ifft(fft_data)
        # print('Restored data after fft for channel %d:' % channel)
        # print(restored_fft_data)

    df = pd.DataFrame(np.transpose(data))
    df.to_pickle("jack_open_trans")

if __name__ == "__main__":
    main()