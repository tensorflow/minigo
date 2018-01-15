

Let's do documentation driven development!


All of the phases of the reinforcement pipeline are driven as subcommands of
main.py

The reinforcement pipeline consists of four commands:

  * selfplay

         -- network: "The path to the network model files",
         -- output-dir: "Where to write the games"="data/selfplay/"
         -- readouts: 'How many simulations to run per move'=0,
         -- games: 'Number of games to play' = 1,
         -- verbose : '>1 will print debug info, >2 will print boards' = 1,

    The selfplay command will play `iters` games between the network whose model
    files are found at path `network`.

    `Readouts` sets how many evaluations the network will use at each move.

    As each game finishes, it will save the training data as DataSet objects to
    `$(output-dir)/$$n.gz`, where
    * `output-dir` is the flag above
    * `timestamp` is the timestamp at which the data was written.
    * `n` is the number of the game played (from the `games` param above)

    Additionally, it will write a metadata file containing the number of positions
    per file to $(output-dir)/$n.meta

  * gather

        -- directory: The directory of data to consolidate into cchunks
        -- output_directory: where to put a training chunk
        -- max_positions: how many positions to accumulate before discarding games
        -- positions_per_chunk: how many positions to collect per chunk.
        -- chunks_to_make: how many times to repeat the sampling process

    We assume the directory given has beneath it the directories of the form

        `$MODEL/$POD/...`

    Given a directory, it...

        -- finds all the .meta files (described in selfplay, above.)
        -- sorts them -- $MODEL is alpha-orderable, so sorting the full path
            should give a list of the most recent games.
        -- adds up the numbers therein until it has a total above 'max_positions'
        -- then, for as many chunks_to_make:
            for range(positions_per_chunk)
                -- pick a random number in the range of max_positions
                -- figure out which game that is by search the cumulative sum of moves.
                -- make a note of the filepath:movenum pair.
            then, sort the list of filepath:movenum pairs, and for each filepath,
            pick out the movenums, slap 'em in a dataset, shuffle the whole thing, and
            write it out to output_directory

    Phew.

  * train

        -- directory: "The path to the training data"
        -- original_model: "the original model to load"
        -- new_model: "Where to save the new model"
        -- epochs=10,
        -- logdir: "where to put the tf logs"

    Given a directory of data, it loads the training chunks found there and
    trains the model located at `original_model` with them, writing the updated
    model to `new_model`

  * evaluate

        -- black_model -- a path to a model,
        -- white_model -- ditto
        -- games: 'Number of games to play' = 400,
        -- readouts: 'How many simulations to run per move'=800,
        -- threshold: 'percent of games challenger must win' = 0.55,
        -- sgf_output: 'where to save the games'

  The evaluator plays two models against each other, the champ and the
  challenger, writing the games out to the directory at `sgf_output`.
  Note the 'threshold' is not arbitrary -- its arrived at by modelling the
  series of games as a bernoulli process and doing MATH.

  Open Questions:
    * How should it communicate the winner?  Re-symlinking the `best_model`
      directory?
    * Robust to stop/resume?
    * what to do with currently running selfplay/trainings as the evaluation
      runs?


So, the overall structure of data for MuGo on the training box is ...
  * `./data/self_play/`
  The datasets for individual games, `$model_name-$playouts-$timestamp.gz`
  * `./data/training_data/`
  The gathered & shuffled datasets for training, grouped into datasets of ~1000 positions,
  e.g. `training_data/$model_name-$playouts-train.gz`
  * `./saved_models/`
  The different models


... and the jobs within the pipeline are

1. The self-play workers, creating new games with the latest version of the
   model.
2. A data preprocessor, that builds data chunks out of the recent games.
3. A trainer, which produces new candidate models.
4. An evaluator, which determines which new candidate model should be
   distributed to the self-play workers for step #1.

These four stages of the pipeline are independent; Kubernetes orchestrates the
jobs on a cluster.  The trainer and preprocessor are run via the `rl_loop`
script, and the self-play workers are treated as a kubernetes Job.  Evaluator
TBD and how it would control things, ditto overall pipeline metrics.



