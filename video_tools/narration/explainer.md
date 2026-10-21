# NEAT Lunar Lander in wind: explainer

## title

What if evolution could design a neural network for you? Not just its weights, but its whole shape. This is NEAT, learning to land on the Moon. And this time, there's wind.

## task

This is Lunar Lander. The goal is to touch down softly between the two flags. Gravity pulls the lander down, and every burst of an engine costs fuel.

## task.2

At every step, the lander senses eight numbers. Where it is, how fast it is moving, how it is tilted and spinning, and whether each leg touches the ground.

## task.3

And at every step, it chooses one of four actions. Do nothing, fire the left engine, fire the main engine, or fire the right engine.

## task.4

And now there is wind. Gusts push the lander sideways, and turbulence twists it, and both keep changing as it flies. The arrow in the corner shows the wind. But the lander can't see that arrow. Wind is not one of its eight inputs. It can only notice that it is drifting, and push back.

## reward

The game hands out points as it goes. Moving toward the pad, slowing down and staying level earn points. Drifting away, speeding up and tilting cost points.

## reward.2

There are also one-off events. Each leg that touches the ground earns ten. Coming to rest earns a hundred. A crash costs a hundred.

## reward.3

And fuel isn't free. Every frame the main engine fires costs a little.

## reward.4

We add one thing of our own. A landing earns a bonus of up to fifty, the full amount exactly in the middle, and nothing at a flag. So evolution is pushed to land in the centre, not just anywhere on the pad.

## reward.5

An average game score of two hundred, without our bonus, counts as solved.

## network

The pilot is a small neural network. The eight inputs come in on the left. They flow along weighted connections, and come out on the right as four scores, one per action.

## network.2

The action with the highest score fires. Watch the scores change as the lander flies.

## network.3

This network has no hidden nodes at all, just twenty-two connections. Nobody designed it. Evolution found every weight, and decided which connections to keep.

## genome

Evolution doesn't work on the network directly. It works on a genome: a list of genes. Node genes say which neurons exist. Connection genes say which neurons are wired together, and with what weight. Read the list, and you can build the network.

## genome.2

Each connection gene also carries a historical marker, called its innovation number. It is given out the first time that connection appears anywhere in the population. So when two genomes share a number, those genes have the same origin. The library we use names each gene by the two nodes it connects, which does the same job.

## mutation

Each new genome starts as a copy of a parent, and then it mutates. There are three kinds of mutation.

## mutation.2

The most common one simply nudges a weight. Here, a positive connection becomes negative.

## mutation.3

The second kind adds a connection, joining two nodes that were not connected before. It gets a new innovation number.

## mutation.4

The third kind adds a node. It splits an existing connection in two. The old gene is switched off, and two new genes take its place. The connection into the new node gets a weight of one, and the one out keeps the old weight, so the network behaves almost the same as before. The structure has grown, without breaking anything.

## crossover

Two parents can also be combined. This is where the innovation numbers pay off. Line the genes up by their numbers, and you can see which genes the two parents share.

## crossover.2

Genes that both parents have are called matching. A gene missing in the middle of the other parent is called disjoint. And a gene beyond the end of the other parent is called excess.

## crossover.3

The child takes each matching gene from either parent at random. The disjoint and excess genes come from the fitter parent. No matter how different the two networks look, the child is always a sensible network.

## speciation

Here is a problem. A mutation that adds a node usually makes the network worse at first, because its new weights are not tuned yet. Ranked against everyone, it would be thrown out before it could ever become useful.

## speciation.2

So NEAT measures how different two genomes are. It counts the genes that don't match, adds the weight differences of the genes that do, and divides by the size of the genome. If that distance is small enough, the two genomes belong to the same species.

## speciation.3

Genomes then compete mostly inside their own species. Each species shares its offspring among its members, so a big species can't crowd out a small new one. And a species that stops improving for twenty generations dies out.

## loop

So here is one generation of NEAT. First, evaluate: every one of the 150 genomes flies, and gets a fitness score.

## loop.2

Then, speciate: sort the genomes into species.

## loop.3

Then, select: inside each species, the best ones become parents, and the two very best are kept unchanged.

## loop.4

And reproduce: crossover and mutation fill the population back up. Then it all starts again.

## fitness

Fitness is the average over five flights of the game score plus the centre bonus.

## fitness.2

Every genome in a generation flies the same starting positions, in the same wind, so the comparison is fair. The starts and the winds change each generation, so no genome can memorise one.

## fitness.3

But five windy flights are noisy, and the top genome is often just lucky. So the ten best genomes of each generation fly thirty more flights before they are kept. And each generation, the champion also flies a hundred flights that no genome ever trained on. We only call the problem solved when that unseen game score reaches two hundred.

## gen_fitness

Now the real run, five hundred generations. Every dot is one genome, and its color shows its species. In generation zero, every network is just the inputs wired straight to the outputs, with random weights. Almost all of them crash.

## gen_fitness.2

Generation by generation, the cloud of dots creeps upward. By generation seventy-four, the champion scores a hundred and fifty on unseen flights.

## gen_fitness.3

The green line is the champion on the hundred unseen flights. It never quite reaches two hundred. The best champion came from generation one hundred and ninety-seven, with a score of a hundred and eighty-seven, and three hundred more generations never beat it. In wind, this problem is still not solved.

## species

This is the classic NEAT picture. Each band is one species, and its height is how many genomes it has. At the start, there is just one.

## species.2

New species split off, grow when they do well, and shrink and die when they stop improving. Over the whole run, thirty-five species came and went.

## gen_ghosts

Here is one generation flying together. All 150 genomes start from the same spot, and we draw each flight as a trail. This is generation zero.

## gen_ghosts.2

At first, nearly every flight ends in a crash.

## gen_ghosts.3

A few dozen generations in, many trails curve down toward the pad. The wind bends them, and they spread out much more than they would in calm air.

## gen_ghosts.4

And late in the run, a share of the population lands from this start, but the trails never bundle into one tidy path. Every gust asks for a slightly different answer. Mutation keeps breaking things, and that is the price of searching.

## growth

This is how the champion's network changed. It started fully connected, with all eight inputs wired to all four outputs, and no hidden nodes.

## growth.2

This time the networks stayed small, because adding new nodes was made rarer. The winner, from generation one hundred and ninety-seven, has no hidden nodes at all, and twenty-two connections. This run was also allowed to grow loops, a kind of memory. The winner has none. More on that in a moment.

## evolution

Here are the champions from different generations, all flying the same windy start.

## evolution.2

The early ones crash or drift away. Later champions fight the drift, slow down, and land between the flags.

## read

So what did evolution build? Let's switch connections off one at a time, and see which ones matter.

## read.2

The most important connection runs from the falling speed to the main engine, with a big negative weight. When the lander falls fast, the main engine's score shoots up, and it brakes. Switch this one connection off, and the lander never lands at all.

## read.3

Next, the tilt angle feeds the left engine. This is one half of what keeps the lander upright. Without it, only twenty percent of these flights land.

## read.4

And the tilt also feeds the right engine, the other way round. That is the other half. Without it, again only twenty percent land.

## ablation

Here are the connections that matter most, ranked by how much the landing rate drops without them.

## ablation.2

Only six of the twenty-two connections can go with almost no loss. This small network uses nearly everything it has.

## ablation.3

And there are no hidden nodes to switch off. Every connection runs straight from a sense to an action. Even in wind, evolution found no use for hidden structure.

## harder

So, is the wind hard enough? Here is the champion from the calm-air version of this project, flying in wind, next to the new winner, on the same start.

## harder.2

In calm air, the old champion landed ninety-five percent of the time, with a score of two hundred and fifty-four. In wind, it drops to sixty-four percent, and a hundred and sixteen. The new winner, trained in wind, lands eighty-one percent, with a score of a hundred and seventy-four, and far more landings in the middle.

## why

So why is it still not solved? Our first wind run had stalled too, so we recorded a thousand of its flights and studied how they fail. Gusts that come and go are fine. But when the wind keeps pushing from one side for most of a flight, the lander fails every single time.

## why.2

And evolution itself was being fooled. Five windy flights are a lucky draw. The genome NEAT picked as best, in orange, always did much worse on a hundred fresh flights, in green. Often a better genome, in blue, was right there in the population, unnoticed. That is why this run gives the top genomes thirty extra flights, and adds new nodes less often.

## memory

The steady push has a deeper cause. Hovering still in calm air, and hovering still against a steady wind, give the network exactly the same eight numbers. But one needs no engine, and the other needs the engine on. A network that only reacts to the moment has to answer both the same way.

## memory.2

Engineers fix this with memory. A loop lets a node keep a running total of the drift, and keep pushing back for as long as the wind lasts. It is the integral part of a PID controller. So this run was allowed to evolve loops, and every network started with a few.

## compare

Here are three runs, flying the same thousand flights. The first run lands sixty-nine percent of them. Fixing the search alone gives seventy-one. The run that was allowed memory lands eighty percent.

## compare.2

But look what happened to the loops. Every genome started with them, and by generation two hundred almost none were left. The winner has no memory at all. So its gain did not come from remembering, and with one run each, we can't yet say how much was luck. None of the runs survive a strong, steady wind.

## outro

On a hundred new windy flights, the winner lands eighty-one percent of the time, thirty-seven percent in the middle, with an average score of a hundred and seventy-four. Not solved, but closer. And it does that with no hidden nodes, and twenty-two connections.

## outro.2

The code, the training run, and these videos are all on GitHub. Clone it, train your own, and watch your network grow.
