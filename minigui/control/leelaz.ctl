board_size = 19

players = {
  "leelaz" : Player("../leela-zero/build/leelaz"
                    " --weights best-network"
                    " --timemanage fast"
                    " -g",
                    startup_gtp_commands=[],
		    cwd="../leela-zero/build"),
}
