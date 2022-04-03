import argparse
from difflib import restore
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
    
    board_id = 38
    params.serial_number = "Muse-748A"

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    time.sleep(120)
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    # demo how to convert it to pandas DF and plot data
    eeg_channels = BoardShim.get_eeg_channels(board_id)
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
    df.to_pickle("Rena_close_3")
    
    # restored_fft = []
    # for count, channel in enumerate(eeg_channels):
    #     # print(f"{channel}!!!!!!!!!!!")
    #     # print('Original data for channel %d:' % channel)
    #     # print(len(data[channel]))
    #     # demo for wavelet transforms
    #     # wavelet_coeffs format is[A(J) D(J) D(J-1) ..... D(1)] where J is decomposition level, A - app coeffs, D - detailed coeffs
    #     # lengths array stores lengths for each block
    #     wavelet_coeffs, lengths = DataFilter.perform_wavelet_transform(data[channel], 'db5', 3)
    #     app_coefs = wavelet_coeffs[0: lengths[0]]
    #     detailed_coeffs_first_block = wavelet_coeffs[lengths[0]: lengths[1]]
    #     # you can do smth with wavelet coeffs here, for example denoising works via thresholds 
    #     # for wavelets coefficients
    #     restored_data = DataFilter.perform_inverse_wavelet_transform((wavelet_coeffs, lengths), data[channel].shape[0],'db5', 3)
    #     # print('Restored data after wavelet transform for channel %d:' % channel)
    #     # print(len(restored_data))

    #     # demo for fft, len of data must be a power of 2
    #     fft_data = DataFilter.perform_fft(data[channel], WindowFunctions.NO_WINDOW.value)
    #     # len of fft_data is N / 2 + 1
    #     restored_fft.append(fft_data)
    #     # restored_fft.append(DataFilter.perform_ifft(fft_data))
    #     # print('Restored data after fft for channel %d:' % channel)
    #     # print(DataFilter.perform_ifft(fft_data))
    
    # restored_fft_data = np.array(restored_fft)
    # print(restored_fft_data.shape)

    # df = pd.DataFrame(np.transpose(restored_fft_data))
    # df.to_pickle("jack_test_trans_fft")

if __name__ == "__main__":
    main()