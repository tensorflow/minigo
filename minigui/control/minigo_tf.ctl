players = {
  "minigo_tf" : Player("bazel-bin/cc/gtp"
                       " --minigui=true"
                       " --engine=tf"
                       " --model=%s"
                       " --num_readouts=%d"
                       " --value_init_penalty=%f"
                       " --courtesy_pass=true"
                       " --virtual_losses=%d"
                       " --resign_threshold=%f" % (
                         FLAGS.model, FLAGS.num_readouts,
			 FLAGS.value_init_penalty, FLAGS.virtual_losses,
			 FLAGS.resign_threshold),
                       startup_gtp_commands=[
                         "report_search_interval 100",
                       ]),
}
