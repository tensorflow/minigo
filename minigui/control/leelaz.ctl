players = {
  # The leelaz path below assumes that the leela-zero root directory lives in
  # the same parent directory as the minigo root. Please modify the leelaz path
  # as required if this is not the case.
  "leelaz" : Player("../leela-zero/build/leelaz"
                    " --weights %s"
                    " --timemanage fast"
                    " -g" % FLAGS.model,
                    startup_gtp_commands=[]),
}
