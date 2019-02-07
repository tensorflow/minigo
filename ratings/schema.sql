/* Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

CREATE TABLE IF NOT EXISTS models (
  id integer primary key,
  model_name text,
  bucket text,
  model_num integer not null,

  num_games integer not null,
  num_wins integer not null,

  ranking float,
  var float,

  black_games integer,
  black_wins integer,
  white_games integer,
  white_wins integer,
  unique(bucket, model_name)
);

CREATE TABLE IF NOT EXISTS wins (
    game_id integer primary key,
    model_winner integer not null,
    model_loser integer not null,
    foreign key(game_id) references games(game_id),
    foreign key(model_winner) references models(id),
    foreign key(model_loser) references models(id)
);

CREATE TABLE IF NOT EXISTS games (
  /* ts-white-model-black-model-number */
  game_id integer primary key,
  timestamp integer not null,
  filename text not null,

  b_id integer not null,
  w_id integer not null,

  black_won boolean,
  result text,
  unique(filename),
  FOREIGN KEY(b_id) REFERENCES models(id),
  FOREIGN KEY(w_id) REFERENCES models(id)
);


CREATE INDEX IF NOT EXISTS model_name_bucket_index ON models (model_name, bucket);
CREATE INDEX IF NOT EXISTS game_model_b_index ON games (b_id);
CREATE INDEX IF NOT EXISTS game_model_w_index ON games (w_id);
CREATE INDEX IF NOT EXISTS game_filename on games (filename);
