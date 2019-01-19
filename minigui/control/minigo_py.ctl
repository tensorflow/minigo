players = {
  "minigo_py" : Player("python"
                       " -u"
                       " gtp.py"
                       " --load_file=%s"
                       " --minigui_mode=true"
                       " --num_readouts=%d"
                       " --conv_width=256"
                       " --resign_threshold=%f"
                       " --verbose=2" % (
                         FLAGS.model, FLAGS.num_readouts,
			 FLAGS.resign_threshold),
                       startup_gtp_commands=[],
		       environ={"BOARD_SIZE": "19"}),
}
