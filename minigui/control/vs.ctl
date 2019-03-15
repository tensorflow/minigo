players = {
  "leelaz" : Player("../leela-zero/build/leelaz"
                    " --weights best-network"
                    " --timemanage off -r 3"
                    " --noponder"
                    " -g",
                    startup_gtp_commands=[
                      "time_settings 0 5 1",
                    ],
                    cwd="../leela-zero/build"),

  "minigo" : Player("bazel-bin/cc/gtp"
                    " --minigui=true"
                    " --model=tf,saved_models/000990-cormorant.pb"
                    " --num_readouts=200"
                    " --value_init_penalty=0"
                    " --courtesy_pass=true"
                    " --virtual_losses=8"
                    " --resign_threshold=-0.8",
                    startup_gtp_commands=[
                      "report_search_interval 100",
                    ]),
}

