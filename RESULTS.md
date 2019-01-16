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
Generations 0-165 played about 12,500 games each, and each model/checkpoint
was trained on about 1M positions (compare to DM's 2M positions in AGZ paper).

After transitioning to a larger board size, and a cluster with GPUs, we started
a new run.  Very quickly, we realized that the batch normalization was incorrect
on the layers of the value head.  After around 500k games, we realized the
policy head should also have had its batchnorm fixed.  This coincided with a
*really* sharp improvement in policy error.

Around model 95, we realized the [initial
Q-value](http://computer-go.org/pipermail/computer-go/2017-December/010555.html)
led to weird ping-ponging behavior inside search, and changed to init to the
average of the parent.

Around model 148, we realized we had the learning rate decay time off by an order of
magnitude...oops.  We also started experimenting with different values for `c_puct`, but didn't really see a difference for our range of values between 0.8 and 2.0.

Around model 165, we changed how many games we tried to play per model, from
12,500 games to 7,000 games, and also increased how many positions each
model was trained on, from 1M to 2M.  This was largely due to how we had
originally read the AGZ paper, where they say they had played 25,000 games with
the most recent model.  However, with 5M games and 700 checkpoints, that works
out to about 7,000 games/checkpoint.  As we were doing more of an 'AZ' approach,
with no evaluation of the new models, we figured this was more accurate.

Around model 230, we disabled resignation in 5% of games and checked our
resign-threshold false positive rate.  We'd been pretty accurate, having tuned
it down to about .9 (i.e., odds of winning at 5%) based on our analysis of true
negatives.  (i.e., what was the distribution of the closest a player got to
resigning who ended up winning)

Around model 250, we started experimenting with tree reuse to gather data about its
effect on the Dirchlet noise, and if it was affecting our position diversity.

We had also been stretching the probabilities pi by raising to 8th power
before training on them, as a compromise between training on the one-hot
vector versus the original MCTS visit distribution.

There's still ambiguity about whether or not a onehot training target for the
policy net is good or bad, but we're leaning towards leaving pi as the softmax
of visits for training, while using argmax as the method for actually choosing
the move for play (after the fuseki).

### Observations

We put the models on KGS and CGOS as 'somebot' starting around version 160.  The
harness running these models is very primitive: they are single-threaded python,
running nodes through the accelerator with a batch-size of ONE... very bad.  The
CGOS harness only manages to run around 300-500 searches per move (5s/move).

However, with that said, we were able to watch it improve and reach a level
where it would consistently lose to the top bots (Zen, AQ, Leela 0.11), and
consistently beat the basic bots, GnuGo, michi, the Aya placeholder bots, etc.
We found CGOS not particularly useful because of this large gap. Additionally,
the large population of LeelaZero variants all had similar playstyles, further
distorting the rating system.

Sometime between v180 and v230, our CGOS results seemed to level off.  The
ladder weakness meant our performance on CGOS was split: either our opponents
could ladder us correctly, or they couldn't :)  As such, it was hard to use CGOS as an
objective measurement.  Similarly, while we were able to notch some wins against
KGS dan players, it was hard to tell if we were still making progress.  This led
to our continued twiddling of settings and general sense of frustration.  The
ladder weakness continued strong all the way to the very end, around model
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


## Third run, Minigo, model 250-..., Jan 20th-Feb 1st (ish)

As our fiddling about after model 250 didn't seem to get us anywhere, and due to our
reconsidering about whether or not to one-hot the policy network target ('pi'), we rolled
back to model 250 to continue training.

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
2. Ambiguity around the training target - should we be trying to predict the MCTS
  visit distribution pi, or the one-hot representation of the final move picked?
  What about in the early game when softpick didn't necessarily pick the move with
  the most visits?
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
   set of holdout positions](https://github.com/tensorflow/minigo/commit/f941f5ac72d860f1f583392cbeb69d0694373824) to use to detect overfitting.
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
   throughput (albeit with some hacks that meant that not every game was sampled
   from equally).
 - General improvements to stackdriver monitoring, like average Q value.
 - Logging MCTS tree summaries in SGFs to inspect tree search behavior.



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

1. during 354 (for 355) change steps per model to 1M from 2M, shuffle
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
and test a new hypothesis:  That our amount of calibration games (5%)
was too small.  Compared to other efforts (AGZ paper says 10%, LZ stepped from 100% => 20%,
and LCZ was still at 100% games played without resignations), Minigo is
definitely the odd one out.

Resign disable rate is a very sensitive parameter; it has several effects:

- Resigning saves compute cycles
- Resigning removes mostly-decided late-game positions from the dataset, helping prevent
  overfitting to a large set of very similar looking, very biased data.
- We need some calibration games; otherwise we forget how to play endgame.
- Calibration games also prevent our bot from going into a self-perpetuating
  loop of "resign early" -> used as training data, resulting in the bot being even
  more pessimistic and resigning earlier.

This, combined with our better selfplay performance, made it a relatively
painless decision to decide to roll back the few days progress, while changing
calibration fraction from 5% -> 20%.

### v7, May 16-July-17

Rolled back to model v5-173; continued with the flags & features added in v7a,
but with holdout data being written from the start ;)  And also with the
calibration rate increased to 20%

We expect to see it go sideways as it adapts to the new proportion of
calibration games, then show a similar improvement to the v7a improvement.
After that, if it flattens off again we'll lower the learning rate further.

After our learning rate cuts, the ratings appeared to taper off.  A brief bit of
"cyclic learning rate" didn't seem to jog it out of its local minima so we
called a halt after ~530 models

v7 also marked the beginning of an auto-pair for the evaluation matches that
would attempt to pair the models about whom our rating algorithm had the
greatest uncertainty.  The best way to increase the confidence of the rating is
to have it play another model with a very close rank, so I wrote it to pick the
model farthest away in time from among the closest rated.  These made for
interesting games, as the models of about the same strength from very different
points in training would take very different approaches.

Our limiting factor continues to be our rate of selfplay.

### v9, TPUS #1, July 19 - August 1

By mid July we were able to get a Cloud TPU implementation working for selfplay and
TPUs were enabled in GKE, allowing us to attempt running with about 600 TPU
v2's.  Also, we were able to run our training job on a TPU as well, meaning that
our training batch size could increase from 256 => 2048, which meant we could
follow the learning rate schedule published in paper #2.

Unfortunately, TPU code required that we change around where we did the rotation
of the training examples, moving it from the host to inside the TF graph.
Unfortunately, I didn't check the default value of the flag and rotating
training examples was turned OFF until around model 115 or so.

Still, this was a hugely successful run:  We were able to run a full 700k steps
-- the length described in Paper 2 and 3 -- in only two weeks.  14M selfplay
games were played; more than the 5M in paper2 and less than the 22M in paper 3.

Here too we saw our top models show up after our first rate cut.  But after the
second rate cut, performance tapered off and the run was stopped.

The top model had a 100% winrate vs our friendly professionals who tried to play
it, which was good to see.  Unfortunately, the model was not able to beat the
best Leela Zero model (from before they mixed in ELF games), so there was
clearly something still amiss.  It was unclear what factors caused the model to
stop improving.  Perhaps not having rotation enabled from the beginning
essentially 'clipped its wings', or perhaps the higher percentage of calibration
games (20%) caused it to see too many useless game positions, or
perhaps we had waited too long to cut the learning rate -- this was the AZ
approach instead of the AGZ approach after all, so perhaps using the AGZ LR
schedule was not appropriate.

In any event, since we could do a full run in two weeks, we could now more
easily forumulate hypotheses and test them.


### v10, TPUS #2, August 28 - Sept 14

Our first set of hypothesis were not ambitious.  Mostly we wanted to prove that
our solution was 'stable' and that we could improve a few of the small things we
knew were wrong with v9, but mostly to leave it the same.  The major differences were:

1. Setting the fraction of calibration games back to 10% (from 20%)
1. Keeping rotation in training on the whole time
1. Playing slightly more games per model
1. Cutting the learning rate the first time as soon as we saw `l2_cost` bottom out and start
   rising, and letting it run at the second rate (0.001) for longer.
1. "Slow window", [described here](https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a),
   a way to quickly drop the initial games from out of the `window` positions
   are drawn from.  Basically, the first games are true noise, so the quicker
   you can get rid of them, the better.

We were very excited about this run.  We had very close fidelity to the
algorithm described in the paper, we had the hyperparameters for batch size and
learning rate as close as we could make them.  Surely this would be the one...
:)

Overall, we played 22M games in 865 models over about two weeks.  The best v10
models were able to pretty convincingly beat the best v9 models, which was
great.  But with all the things we changed, which was responsible?

In retrospect, probably the two most useful changes were leaving the learning
rate at the `medium` setting for longer, giving it a longer time to meander
through the loss landscape, and setting the calibration rate back to 10%.
(Having rotation on in training the whole time obviously doesn't hurt, but it
probably doesn't affect the overall skill ceiling)

The theory on having the resign-disable games correctly set was based off of
some of the [excellent results by Seth on the sensitivity of the value network to
false-positive resigns](https://github.com/tensorflow/minigo/issues/483).  His
analysis was to basically randomly flip the result of x% of training examples
and measure the effects on value confidence and value error.  The observation
was that having 5% of the data fouled was sufficient to dramatically affect
value error.

With this amount as a good proxy for 'the amount of data needed to impact
training significantly', we could then think about how the percent of
calibration games might be affecting us.  With 20% of the
games being played to scoring, sometimes requiring 2x or 3x as many moves, it
meant that the moves sampled from these games were essentially 'useless' i.e.,
from a point beyond which the outcome was no longer in doubt.  Since we were
still sampling uniformly by game (i.e., picking 4 moves from the most recent
500k games), this meant that for v9, exactly 20% of the training examples were
from these games.  Reducing that to 10% probably gave us an edge in establishing
a better value net.

It seemed like there was no real difference from playing more games per model,
and 'slow window' doesn't seem to dramatically affect how long it takes to get
out of the early learning phase.  (Basically, policy improves while value
cost drops to 1.0, implying pure randomness, until eventually the net starts to
figure out what 'causes' wins and value error drops below 1.0)

v10 showed that we could reach about the same level as v9, and that we'd plateau
after a certain point.  Also, we noted that the variance in strength between
successive networks was really high!  It seemed to expand as training continued,
suggesting that we were in some very reliable minima.  Was this a function of the higher batch
size, the higher number of games per model, or both, or neither?

Qualitatively, v10's Go was interesting for a couple reasons.  We saw the
familiar 'flower' joseki that was the zero-bot signature, but we also saw the
knight's move approach to the 4-4 point come into favor.  However, by the end of
the run, the bot was playing 3-4 points as black, leading to very different
patterns than those seen by AGZ or AG-Master.

So while we (again) created a professional level go AI, we were still below the
level of other open-source bots (ELF and LZ), despite the fact that the
AlphaZero approach was supposed to be equal or slightly superior.  Why?  What
were we missing?

### v11, Q=0, Sept 14- Sept 17

Since we were rapidly running out of ways we were differing from the paper, we
thought we'd test a pretty radical hypothesis, called "Q=0".  The short version
is that, per the paper, a leaf with no visits should have the value initialized
to '0'.  Say a board position had an average Q-value of 0.2, suggesting that it
slightly favored black.  Suppose further that, after one move, the expected
result of the board was not dramatically different, i.e., it would evaluate to
near 0.2 for most reasonable moves.  If it were white to play, picking the leaf
with the value closest to white's goal (-1) would mean search would prefer to
choose any unvisited leaf (i.e., leaves still with their initialized value of
0).  This would result in search visiting all legal moves before returning to
the 'reasonable' variaitions.  The net result would mean a huge spread of
explored moves for the losing side, and a very diluted search.

Conversely, if it were black to play, search would prefer to only pick visited
leaves, as unvisited leaves would have a value of 0, while any reasonable first
move would have a value presumably close to 0.2, which is closer to black's goal
(+1).  The result here is that search would basically only explore a single move for the
winning side.

The results would be policy examples that are very flat or diffuse for the side that
loses more, and very sharp for the side that wins more.

We had run all of our previous versions assuming that instead of Q=0, we should
initialize leaves to the Q of their parent, aka 'Q=parent'.  This seems reasonable:  both sides
would look for moves that represented improvements from a given position, and
the UCB algorithm could widen or narrow the search as needed.

But what if we were wrong?  What if exhaustively checking all moves for the
losing side and essentially one-hot focusing on the moves for the winning side
was an intentional choice?  It's worth pointing out that this was one of the
first points of contention found by computer-go researchers between the release
of the second paper and before the release of the third.  You can see some
inscrutable replies and a lot of head scratching on the [venerable computer-go
mailing list here](http://computer-go.org/pipermail/computer-go/2017-December/010550.html).
Even with a direct reply from a main author, opinions were still divided as to
what it meant.

So, we set Q=0 and ran V11.  Our expectation was that it would fail quickly in
obvious ways and we could put this speculation to bed.  We were half-right...

After a few hundred models, where strength did not increase as quickly as with
Q=parent, the resign-rate false-positive began to swing wildly out of control.
It seemed that as a side began to win more than 50% of its games, search began
to greatly favor its policy, and the losing side was basically always left with
a handicap of having a massive number of its moves 'diluted.'  Continuing from
our previous example... black would be able to spread out its reads quicker than
white would be able to focus on a single useful line, essentially leaving one
side at a disadvantage.  

Compounding the problem, our pipeline architecture meant that the
'false-positive' resign rate appeared as a lagging indicator: a new model would be
released, the TPU workers would all start playing with it immediately, and the
90% of games with resign enabled would finish before the 10% of holdout games
that could be measured to set the threshold... So we were right that this failed
quickly.

Where we were wrong was that this would put the theory to bed.  Arguing against
Q=0 was the fact that it was clearly unstable, has problems with the AZ method,
and intuitively seemed to massively disadvantage one side.  Arguing for it was
the fact that it was actually what was published, that there was a plausible
explanation for the imbalance, and that the pipeline could be improved or
tweaked to only play the resign-holdout games first.  But most interestingly was
that, despite the very different behavior of selfplay and the different dynamics
showing on tensorboard, the performance of the models on the professional player
validation set was not awful!  It still improved, and it improved fairly well --
although slower than it improved with Q=parent.

So perhaps with a better pipeline and tighter controls on the resign threshold,
this would ultimately lead to better exploration and a better model, but in the
meantime it looked far more unstable.  We stopped the run after 3 days.

### v12, vlosses 2

With Q=0 checked off our list, the next question mark for us was "virtual
losses".  In brief, virtual losses provide a way for GPUs to more efficiently do
tree search by selecting multiple leaves.  Simply running search repeatedly will
almost always result in the same leaf being chosen, so to fill a 'batch' of
leaves, chosen leaves are given a 'virtual loss', encouraging search to pick a
different variation to explore the next time down the tree.  Once the batch is
full, the GPU can efficiently evaluate them all, the 'virtual losses' are
reverted and the real results from the value network are propagated up the tree.
The number of simultaneous nodes would be the batch size, and we set it via the
'vlosses' parameter.  Our previous runs all used vlosses=8.  We had
only one sentence from the second paper to guide us: "We use virtual loss to
ensure each thread evaluates different nodes" (from Methods, "Search Algorithm",
*Backup*)

In particular, the cost of running a neural network is greatly amortized by the
highly parallel nature of GPUs and TPUs.  E.g., the time taken to evaluate one
node might be nearly the same as the cost of evaluating two nodes, as the
overhead of transferring data to/from the GPU/TPU is incurred regardless of how
much data is sent.

For TPUs, which really shine with high batch sizes, virtual losses seemed like
a necessity.  But how would they affect reinforcement learning?  On the plus
side, virtual losses acted almost like increased noise, encouraging exploration
of moves that the network might not have chosen in favor of a single main line.
On the minus side, MCTS' mathematically ideal batch size was 1: Evaluating many
sub-optimal moves might distort the averages, particularly in the case of long,
unbranching sequences with a sharp horizon effect, i.e., ladders.  Both v9 and
v10, while reaching professional strength, would still struggle with ladders.

In a ladder sequence, only one node in the MCTS tree would read the ladder,
while the rest of the batch would need to pick non-ladder moves.  It's
conceivable that the averaging done by MCTS would essentially encourage the
models to avoid sharp, narrow paths with high expected value in favor of flatter paths with a less dramatic outcome.  We figured we'd test this by setting batch size to 2, doubling the number of games played at once (to achieve roughly the same throughput), and run again.

Early results are not dramatically stronger.  Qualitatively, we have not seen
the model explore the knight's move approach to the 4-4 hardly at all, instead
immediately converging to the 3-4 openings as black followed by an immediate
invasion of the 3-3 point.

Having learned from our previous runs, we postponed our final rate cut until
step 800k or so, and with it we also increased our window size to 1M (assuming
that the lower rate of change between models would lead to a smaller diversity
of of selfplay game examples).

One other minor tweak was to stop training on moves after move #400 after we had
reached a reasonable strength.  The "logic" here is similar to v10's explanation
re: the 10% calibration games.  if the calibration games run, on average,
2x or 3x longer, then the examples taken from them are almost always past the
point when the outcome is well understood.  So forcibly drawing the training
examples from the first 400 moves should mostly rid us of those pointless
examples without dramatically distorting the ratio of opening/middle/endgame
examples seen by the network.

This run ended up not really having fixed anything. On our cross-evaluation 
matches, v12 ended up slightly stronger than the other runs, but it was well
within the natural variability of the other runs. V12 still had ladder issues
and it still could not beat LeelaZero or ELF, even with increased playouts.

In retrospect, the fact that v12 had improved slightly in strength suggested
that adding spurious reads to the game tree would diminish MCTS's ability to
discern the best move.


### v13, pro game bootstrap

Out of curiosity, we wanted to see what happened if we bootstrapped a model
from a pro game dataset. The expected thing happened: the model started from
a much stronger starting place, and the distribution of opening moves played
was very reminiscent of human pro play. We eventually saw the model discard
some common human opening patterns and revert back to Zero-ish play (e.g.
lots of 3-3 invasions). Overall, the run went well, but the strength never
peaked to the same level as v10 or v12.

### v14, Bigtable integration with init-to-loss switch

For v14, the biggest change was replacing the in-memory TFExample shuffler with
a BigTable shuffler.

A minor change - we started logging the average depth of search.

Previously, we'd write out tiny TFRecords containing ~200
TFExamples each, and another Python process would rsync the records to one
machine, read them into memory, and maintain a large moving window of positions
in memory that would be periodically be spit out as training chunks. This shuffler
had all sorts of hacks in it:

  - we'd entirely ignore any games with < 30 moves
  - at the end of v12, we tried ignoring moves > 400 to remove post-endgame noise
  - we'd draw a fixed number (4) of moves from each game (which oversampled
    short games). This hacky fixed-number sampling was because exactly 10% of
    our games were calibration games, but those calibration games were much 
    longer than the other games, meaning that an even sampling by move would
    oversample the calibration games. A fixed-number sampling would at least
    guarantee a 10% representation of calibration games.

In terms of bottlenecks, our sampler took a big chunk of time to run, but it
was about the same as training time, so it was not technically a bottleneck
yet. But we were still training on a single TPU (8 cores) rather than a pod
(128 cores), so if we were to scale to more TPUs for selfplay, and upgrade our
trainer to use a pod to compensate, then this shuffler would quickly become the
bottleneck.

For v14, our selfplay workers would instead directly upload each TFExample
to BigTable, indexed by game number and move number. A set of parallel readers
would then use BigTable's built-in samplers to read over a specified window of
games. In order to ensure a 10% ratio of calibration games to resign-enabled
games, we used the BigTable input readers to enforce a 90:10 ratio.

The many benefits of BT replacement included:

  - we could directly run a query on BigTable to compute the appropriate resign
  threshold. Previously, this had been done by manually inspecting a
  Stackdriver graph and editing a flagfile on GCS. This meant that the latency
  of resign threshold updates dropped from hours to minutes. Also, because
  the resign threshold had been set manually, we tended to pick a number on the
  high side, since it would be in force for a while.
  - Time between checkpoints dropped from 30 minutes to 20 minutes. This
  was due to the trainer not having to wait for the shuffler at all. We actually
  ended up adding waiting logic to the timer to ensure enough games had been
  completed before starting training again. This was probably an improvement on
  our previous logic, which output a golden chunk as soon as the trainer completed
  training a new generation.
  - The ratio of sampled positions from calibration games and normal games was
  precisely set at 90:10, instead of implicitly depending on the ratio of game
  lengths for calibration games / normal games. (The AGZ paper reported setting
  aside 10% of games, which could lead to anywhere from 10~20% of moves being
  from calibration games.

The run was uneventful and was a bit better than previous runs; it escaped random
play much more quickly, probably because the up-to-date resign thresholds
meant that our training data was appropriately pruned of 'easy' positions.
So all in all, the rewrite didn't create any new pipeline issues and improved
many aspects.

Around model 280, we [discovered on LCZero forums](http://talkchess.com/forum3/viewtopic.php?f=2&t=69175&p=781765&sid=c57776201e233b1be14bf56f71f5e54e#p781765)
that AlphaGoZero had used
init-to-loss for Q values. We switched to init-to-loss from init-to-parent and
saw our network gain dramatically in strength. Soon, our top model thoroughly
exceeded our previous best model, and beat LZ's v157, the last model from before
LZ started mixing in game data from the much stronger ELF.

### v15, clean run with init to loss

This run beat v14 pretty early on, producing a model that was stronger than v14's
best at model 330. v15 ended up slightly stronger than v14 but was otherwise
similar.

Qualitatively, we noticed that V14/V15's models were much sharper,
meaning that they had more concentrated policy network output and the value
network output would have stronger opinions on whether it was winning or losing.
Simultaneously, the calibrated resign threshold for init-to-loss ended up at
0.5-0.7 compared to 0.8-0.95 in previous runs. So v14/v15's MCTS was operating
in the sweet spot of [-0.5, 0.5] more frequently. (The MCTS algorithm tends to
go blind once Q gets close to 0.9, because there's no more room at the top.)

The init-to-loss configuration also led to much deeper reads, as the tree search
behavior would dive deeply into one variation before going onto the next.
Originally, we'd labelled this behavior as pathological and thought it went
against the spirit of MCTS, which was why init-to-loss wasn't seriously
considered. But this read behavior meant that v14 was our first model to be
able to consistently read ladders. One funny side effect was that
it was actually not clear whether v14 was actually stronger than v12, or whether
v14 was just bullying v12 around by reading ladders that v12 couldn't read, and
winning easy rating points. But since we had achieved success against external
reference points, we concluded that v14/v15 were in fact really stronger than v12.

During v15's run, we realized that our evaluation cluster infrastructure was in
need of some upgrades, primarily for two reasons:

1. We wanted to include external reference points, so that we could get a more
stable evaluation.
1. We wanted to be able to run each model with its preferred configuration (init
to parent for v10/v12, and init to loss for v14/v15.) This would seem easy - run
each side as an independent subprocess. However, because of GPU memory / TPU
contention issues, this was tricky to do.
