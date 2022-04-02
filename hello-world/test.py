# Code example for logging data from the Muse 2
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

def main():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = "COM port for BLED112 Dongle"
    params.timeout = "timeout for device discovery or connection"
    board = BoardShim(22, params)
    board.prepare_session()
    board.start_stream() # use this for default options
    time.sleep(10)
    data = board.get_board_data()  # get all data and remove it from internal buffer
    board.stop_stream()
    board.release_session()
    print(data)

if __name__ == "__main__":
    main()