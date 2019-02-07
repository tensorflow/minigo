competition_type = 'playoff'
description = """Example control file"""

board_size = 19
komi = 7.5

record_games = True
stderr_to_log = True

def LeelaPlayer(model, playouts):
    return Player(
        "./leelaz -g --noponder -w {} -t 1 -p {} --timemanage off ".format(
            model, playouts),
        startup_gtp_commands=["time_settings 0 1 0"])

matchups = []
players = {}
for playouts in ["10000"]:
  p1 = LeelaPlayer('lz202.gz', playouts)
  p2 = LeelaPlayer('mg-16-833.gz', playouts)
  p1_name = "lz202_p{}".format(playouts)
  p2_name = "mg-16-833{}".format(playouts)
  players[p1_name] = p1
  players[p2_name] = p2
  matchup_name = "lz202_vs_mg-16-833_p{}".format(playouts)
  matchups.append(Matchup(
     p1_name, p2_name, id=matchup_name, number_of_games=2,
     alternating=True, scorer='players'))
