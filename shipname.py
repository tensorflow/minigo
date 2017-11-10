
import random

SHIP_FILE="data/names.txt"

names = open(SHIP_FILE).read().split('\n')

def generate():
    return random.choice(names)
