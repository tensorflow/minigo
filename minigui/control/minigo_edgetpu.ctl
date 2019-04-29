board_size = 19

players = {
  "minigo_edgetpu" : Player("python"
                            " -u"
                            " gtp.py"
                            " --load_file=saved_models/v17-2019-04-29-edgetpu.tflite"
                            " --minigui_mode=true"
                            " --num_readouts=800"
                            " --resign_threshold=-0.8"
                            " --parallel_readouts=1"
                            " --verbose=2",
                            startup_gtp_commands=[],
                            environ={"BOARD_SIZE": str(board_size)}),
}
