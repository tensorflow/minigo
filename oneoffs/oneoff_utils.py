import numpy as np
import os
import pandas as pd
import sgf
import utils
import utils

from tqdm import tqdm
from sgf_wrapper import sgf_prop, replay_sgf


def get_sgf_props(sgf_path):
  with open(sgf_path) as f:
    sgf_contents = f.read()
  collection = sgf.parse(sgf_contents)
  game = collection.children[0]
  props = game.root.properties
  return props


def parse_sgf(sgf_path):
  with open(sgf_path) as f:
    sgf_contents = f.read()

  collection = sgf.parse(sgf_contents)
  game = collection.children[0]
  props = game.root.properties
  assert int(sgf_prop(props.get('GM', ['1']))) == 1, "Not a Go SGF!"

  result = utils.parse_game_result(sgf_prop(props.get('RE')))

  positions, moves = zip(*[(p.position, p.next_move) for p in replay_sgf(sgf_contents)])
  return positions, moves, result, props

def check_year(props, year):
  if year is None:
    return True
  if props.get('DT') is None:
    return False

  try:
    #Most sgf files in this database have dates of the form
    #"2005-01-15", but there are some rare exceptions like
    #"Broadcasted on 2005-01-15.
    year_sgf = int(props.get('DT')[0][:4])
  except:
    return False
  return year_sgf >= year


def check_komi(props, komi_str):
  if komi_str is None:
    return True
  if props.get('KM') is None:
    return False
  return props.get('KM')[0] == komi_str


def find_and_filter_sgf_files(base_dir, min_year = None, komi = None):
  sgf_files = []
  count = 0
  print("Finding all sgf files in {} with year >= {} and komi = {}".format(base_dir, min_year, komi))
  for i, (dirpath, dirnames, filenames) in tqdm(enumerate(os.walk(base_dir))):
    for filename in filenames:
      count+=1
      if count%5000 == 0:
        print("Parsed {}, Found {}".format(count, len(sgf_files)))
      if filename.endswith('.sgf'):
        path = os.path.join(dirpath, filename)
        props = get_sgf_props(path)
        if check_year(props, min_year) and check_komi(props, komi):
          sgf_files.append(path)
  print("Found {} sgf files matching filters".format(len(sgf_files)))
  return sgf_files

