board_size = 19

players = {
  "minigo_tf" : Player("bazel-bin/cc/gtp"
                       " --minigui=true"
                       " --model=tf,saved_models/000990-cormorant.pb"
                       " --num_readouts=64"
                       " --value_init_penalty=0"
                       " --courtesy_pass=true"
                       " --virtual_losses=8"
                       " --resign_threshold=-0.8",
                       startup_gtp_commands=[
                         "report_search_interval 100",
                       ]),
}
