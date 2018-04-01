package(
    default_visibility = ["//visibility:public"],
)

# Build condition label that matches when C++ Minigo is being built for a 9x9
# board.
# Used by the minigo_9_test build rule.
config_setting(
    name = "minigo9",
    values = {"copt": "-DMINIGO_BOARD_SIZE=9"},
)

# Build condition label that matches when C++ Minigo is being built for a 9x9
# board.
# Used by the minigo_19_test build rule.
config_setting(
    name = "minigo19",
    values = {"copt": "-DMINIGO_BOARD_SIZE=19"},
)
