# Results

## First run, 9x9, November 2017.

 - Ran on a cluster of ~1k cores, no GPUs, 400 readouts/move, 64 games played in
   parallel per worker.  Trained on a single P100 machine.
   - No tree re-use
   - No 'resign disabled' holdout
   - No vectorized updates
   - Dir noise probably parameterized wrong
   - `c_puct` a guess at 2.5
   - "Stretch" for moves after the opening
   - No evaluation of new networks
   - [Init Q to 0]( http://computer-go.org/pipermail/computer-go/2017-December/010555.html)
   - Hardcoded not to pass early on (before move 20 i think).
 - Ran for two weeks
 - Evaluated by playing a variety of kyu & dan players, judged to be SDK

This first run was largely intended to confirm that the network and MCTS code
worked and that the algorithm as described would improve.  As such, the CPU-only
version on the smaller board was a modest success.  Even without a number of
enhancements described in the paper (re-use, etc), play was observed to improve
rapidly towards reasonable human play.

We didn't see any 'ladder weakness', probably due to the smaller board size.

What's really shocking is how effectively this worked even with many parts of
the code that were absolutely wrong.  In particular, the batch normalization
calls (`tf.layers.batch_normalization`) were entirely wrong, and the lack of
correct batch normalization almost certainly slowed down progress by an order of
magnitude.

As a result, it's hard to say how effective it was at proving things were
correct.  The results were good enough to assume that everything would work fine
when scaled up to 19x19, but obviously that wasn't true.  As a result, it was
hard to experiment on 9x9s -- If it worked on 9x9s when broken, who was to say
that "working on 9s" was useful or predictive?

Run was stopped as a result of internal resource allocation discussions.

## Second run, 'somebot', 19x19, Dec'17-Jan'18

This second run took place over about 4 weeks, on a 20 block network with 128
filters.  With 4 weeks to look at it and adjust things, we didn't end up keeping
the code constant the whole time through, and instead we were constantly
monkeying with the code.

This ran on a Google Container Engine cluster of pre-release preemptible GPUs;
(TODO: link) Hopefully the product was improved by our bug reports :)

### Adjustments
Generations 0-165 played about 12,500 games each, and each generation/checkpoint
was trained on about 1M positions (compare to DM's 2M positions in AGZ paper).

After transitioning to a larger board size, and a cluster with GPUs, we started
a new run.  Very quickly, we realized that the batch normalization was incorrect
on the layers of the value head.  After around 500k games, we realized the
policy head should also have had its batchnorm fixed.  This coincided with a
*really* sharp improvement in policy error.

Around generation 95, we realized the [initial
Q-value](http://computer-go.org/pipermail/computer-go/2017-December/010555.html)
led to weird ping-ponging behavior inside search, and changed to init to the
average of the parent.

Around v148, we realized we had the learning rate decay time off by an order of
magnitude...oops.  We also started experimenting with different values for `c_puct`, but didn't really see a difference for our range of values between 0.8 and 2.0.

Around v165, we changed how many games we tried to play per generation, from
12,500 games to 7,000 games, and also increased how many positions each
generation was trained on, from 1M to 2M.  This was largely due to how we had
originally read the AGZ paper, where they say they had played 25,000 games with
the most recent model.  However, with 5M games and 700 checkpoints, that works
out to about 7,000 games/checkpoint.  As we were doing more of an 'AZ' approach,
with no evaluation of the new models, we figured this was more accurate.

Around v230, we disabled resignation in 5% of games and checked our
resign-threshold false positive rate.  We'd been pretty accurate, having tuned
it down to about .9 (i.e., odds of winning at 5%) based on our analysis of true
negatives.  (i.e., what was the distribution of the closest a player got to
resigning who ended up winning)

Around v250, we started experimenting with tree reuse to gather data about its
effect on the Dirchlet noise, and if it was affecting our position diversity.
We also changed from 'stretching' the probabilities pi by exponentiating to a
factor of 8 (i.e., temperature 0.125), to making it onehot (infinitesimal
temperature), to leaving it proportional to visits (unity temperature).

There's still ambiguity about whether or not a onehot training target for the
policy net is good or bad, but we're leaning towards leaving pi as the softmax
of visits for training, while using argmax as the method for actually choosing
the move for play (after the fuseki).

### Observations

We put the models on KGS and CGOS as 'somebot' starting around version 160.  The
harness running these models is very primitive: they are single-threaded python,
running nodes through the accelerator with a batch-size of ONE... very bad.  The
CGOS harness only manages to run around 300-500 searches per move.

However, with that said, we were able to watch it improve and reach a level
where it would consistently lose to the top bots (Zen, AQ, Leela 0.11), and
consistently beat the basic bots, GnuGo, michi, the Aya placeholder bots, etc.

Sometime between v180 and v230, our CGOS results seemed to level off.  The
ladder weakness meant our performance on CGOS was split: either our opponents
could ladder us correctly, or they couldn't :)  As such, it was hard to use CGOS as an
objective measurement.  Similarly, while we were able to notch some wins against
KGS dan players, it was hard to tell if we were still making progress.  This led
to our continued twiddling of settings and general sense of frustration.  The
ladder weakness continued strong all the way to the very end, around generation
v280, when our prerelease cluster expired.

On the whole, we did feel pretty good about having answered some of the
ambiguities left by the papers:  What was a good value for `c_puct`?  Was
tree-reuse for evaluation only, or also for selfplay?  What was up with
initializing a child's Q to zero?  How about whether or not to one-hot the
policy target pi?

Worth observing:  For nearly all of the generations, 3-3 was its ONLY preferred
opening move.  This drove a lot of the questions around whether or not we had
adequate move diversity, sufficient noise mixing, etc.  It is unclear whether
the early errors (e.g., initial 50-100 generations with broken batch-norm) have
left us in a local minima with 3-3s that D-noise & etc. are not able to
overcome.


## Third run, Minigo, v250-..., Jan 20th-Feb 1st (ish)

As our fiddling about after v250 didn't seem to get us anywhere, and due to our
reconsidering about whether or not to one-hot the policy network target ('pi'), we rolled
back to version 250 to continue training.

We also completely replaced our custom class for handling training examples
(helpfully called `Dataset`, and predating tf.data.Dataset) with a proper
implementation using tfExamples and living natively on GCS.  This greatly
simplified the training pipeline, with a few hiccups at the beginning where the
training data was not adequately diversified.

The remaining differences between this run and the described paper:

1. No "virtual losses"
  - Virtual losses provide a way to parallelize MCTS and choose more nodes for
    parallel evaluation by the accelerator.  Instead of this, we parallelize by
    playing multiple games per-worker.  The overall throughput is the same, in
    terms of games-per-hour, but we could increase the speed at which games are
    finished and thus the overall recency of the training data if we implemented
    virtual losses.
  - A side effect of virtual losses is that the MCTS phase is not always
    evaluating the optimal leaf, but is evaluating instead its top-8 or top-16
    or whatever the batch size is.  This has a consequence of broadening the
    overall search -- is this broadening important to the discovery of new moves
    by the policy network, and thus the improvement of the network overall?  Who
    knows? :)
2. Ambiguity around the transformation of the MCTS output, pi.
3. Too few filters (128 vs 256).


Results: After a few days a few troublesome signs appeared.
  - Value error dropped far lower than the 0.6 steady-state we had expected from
    the paper. (0.6 equates to ~85% prediction accuracy).  This suggested our
    new dataset code was not shuffling adequately, leading to value overfitting
  - This idea was supported when, upon further digging, it appeared that the
    network would judge transformations (rotation/reflection) of the *same
    board* to have wildly different value estimates.

These could also have been exacerbated by our computed value for the learning
rate decay schedule being incorrect.

This run was halted after discovering a pretty major bug in our huge Dataset
rewrite that left it training towards what was effectively noise.  Read the gory
details
[here](https://github.com/tensorflow/minigo/commit/9c6e013293d415e90b92097327dfaca94a81a6da).

Our [milestones](https://github.com/tensorflow/minigo/milestone/1?closed=1) to hit before running a new 19x19 pipeline:
  - add random transformations to the training pipeline
  - More completely shuffle our training data.
  - Figure out an equivalent learning rate decay schedule to the one described
    in the paper.


### Fourth run, 9x9s.  Feb 7-March

Since our 19x19 run seemed irrepairable, we decided to work on instrumentation
to better understand when things were going off the rails while it would still
be possible to make adjustments.

These [improvements](https://github.com/tensorflow/minigo/milestone/2?closed=1) included:
 - Logging the [magnitude of the
   updates](https://github.com/tensorflow/minigo/commit/360e056f218833938d845b454b4e24158034b58a)
   that training would make to our weights.
 - [Setting aside a
   set of positions](https://github.com/tensorflow/minigo/commit/f941f5ac72d860f1f583392cbeb69d0694373824) to use to detect
overfitting.
 - Improvements to setting up clusters and setting up automated testing/CI.


The results were very promising, reaching pro strength after about a week.

Our models and training data can be found in [the GCS bucket
here](https://console.cloud.google.com/storage/browser/minigo-pub/v3-9x9).


### V5, 19x19s, March-April


With our newly improved cluster tools, better monitoring, etc, we were pretty
optimistic about our next attempt at 19x19.  Our evaluation matches -- pitting
different models against each other to get a good ratings curve -- were
automated during the v5 run, and a lot of new data analysis was made available
during the run on the 'unofficial data site'
['cloudygo'](http://www.cloudygo.com)

Unfortunately, progress stalled shortly after cutting the learning rate, and
seemed to never recover.  Our three most useful indicators were our value net's
train error and validation error, our value net error on a set of professional
games (aka "[figure 3](http://cloudygo.com/v7-19x19/figure-three)"), and our selfplay rating as measured by our evaluation
matches.

For these measures, shortly after our learning rate cut, performance improved
dramatically before reverting almost completely, with the value net eventually
becoming overfit to be worse than random chance on the holdout data.

Various explanations for how the training data could be qualitatively different
than the holdout data to result in overfitting were advanced:  Inadequate
[shuffling of examples](http://www.moderndescartes.com/essays/shuffle_viz/), a
too aggressive learning rate, not enough 'new games' played by new models before
training new models, etc.

After it seemed clear that it was not going to recover, the last half of the run
(after 315 or so) was largely treated as a free time to twiddle knobs and
observe their effects to try and narrow down these effects.

During v5, we wrote a number of new features, including:

 - A port of the engine to C++ for better performance.
 - A new UI (in the minigui/ directory)
 - Automated evaluation on a separate cluster
 - Separate tfexample pipeline guaranteeing complete shuffling and better
   throughput.
 - And general improvements to our monitoring and data analysis.


#### v5 changelog:

1. 3/4 of the way through 125, change squash from 0.95 => 0.98, change temp cutoff
to move 30 (from 31, because odd number = bias against black)

1. 192 -- learning rate erroneously cut to 0.001.  207 returned to 0.01.
(206 trained with 0.001, 207 @0.01)  (this was eventually reverted and moved off
as v6)

1. 231 -- moved #readouts to 900 from 800.

1. 295-6 -- move shuffle buffer to 1M from 200k

1. 347 -- changed filter amount to 0.02.  (first present for 347)

1. 348 reverted shuffle buffer change

1. 352 change filter to 0.03 and move shuffle buffer back to 200k

1. during 354 (for 355) change steps per generation to 1M from 2M, shuffle
buffer down to 100k

1. 23.2M steps -- change l2_strength to 0.0002 (from 0.0001)

1. 360ish -- entered experimental mode; freely adjusted learning rate up and
down, adjusted batch size, etc, to see if valnet divergence could be fixed.


### v7a, first week of May

We began our next run by rolling back to v5-173-golden-horse, chosen because of
its relatively high performance and it seemed to be before the v5 run flattened
out.

The major changes were:

 - using the c++ selfplay worker
 - better training data marshalling/shuffling
 - larger training batch size (16 => 256)
 - lower learning rate (1e-2 => 1e-3) (note this tracks the original aborted LR
   cut at model #192 in the short-lived 'v6')
 - higher numbers of games per model ( >15k, usually ending up with ~20k)


v7's first premature run had a couple problems:

 - We weren't writing our holdout data for 174-179ish, resulting in some weird tensorboard
   artifacts as our holdout set became composed of models farther and farther
   from the training data
 - We also weren't writing SGFs, making analysis hard.
 - But we did have our 'figure three', so we were able to make sure we were
   moving in the right direction :)

But it did make very strong progress on selfplay!  And then stopped...why?

After finding a [potentially bad engine
bug](https://github.com/tensorflow/minigo/pull/234) we decided to pause training
to get to the bottom of it.  It ended up affecting < 0.01% of games, but we had
already seen progress stall and retrace similar to v5, so we decided to rollback
and test a new hypothesis:  That our amount of games with resign disabled (5%)
was too small.  Compared to other efforts (AGZ paper says 10%, LZ stepped from 100% => 20%,
and LCZ was still at 100% games played without resignations), Minigo is
definitely the odd one out.

This, combined with our better selfplay performance, made it a relatively
painless decision to decide to roll back the few days progress.

### v7, May 16-

Rolled back to model v5-173; continued with the flags & features added in v7a,
but with holdout data being written from the start ;)  And also with the
resign-disabled fraction increased to 20%

We expect to see it go sideways as it adapts to the new proportion of
resign-disabled games, then show a similar improvement to the v7a improvement.
After that, if it flattens off again we'll lower the learning rate further.

Our limiting factor continues to be our rate of selfplay.

