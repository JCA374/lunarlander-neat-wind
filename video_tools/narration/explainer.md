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

This network has just one hidden node, and twenty-six connections. Nobody designed it. Evolution found every weight, and decided which connections to keep.

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

But five flights are noisy, and in wind they are very noisy. The top genome is often just lucky. So each generation, its champion also flies a hundred flights that no genome ever trained on. We only call the problem solved when that unseen game score reaches two hundred.

## gen_fitness

Now the real run, five hundred generations. Every dot is one genome, and its color shows its species. In generation zero, every network is just the inputs wired straight to the outputs, with random weights. Almost all of them crash.

## gen_fitness.2

The cloud of dots rises quickly at first. By generation forty-two, the champion scores a hundred and fifty on unseen flights.

## gen_fitness.3

But then it stalls. The green line is the champion on the hundred unseen flights. It never reaches two hundred. The best champion of the whole run came early, from generation eighty-one, with a score of a hundred and eighty-two. The remaining four hundred generations never beat it. In wind, this problem was not solved.

## species

This is the classic NEAT picture. Each band is one species, and its height is how many genomes it has. At the start, there is just one.

## species.2

New species split off, grow when they do well, and shrink and die when they stop improving. Over the whole run, forty-six species came and went.

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

Did the wind make hidden nodes necessary? The answer is: not really. The winner, from generation eighty-one, has a single hidden node. After that, the networks kept growing, to more than thirty hidden nodes and eighty connections. But those big networks flew worse on unseen flights, not better. More structure did not buy more skill here.

## evolution

Here are the champions from different generations, all flying the same windy start.

## evolution.2

The early ones crash or drift away. Later champions fight the drift, slow down, and land between the flags.

## read

So what did evolution build? Let's switch connections off one at a time, and see which ones matter.

## read.2

The most important connection runs from the falling speed to the main engine, with a big negative weight. When the lander falls fast, the main engine's score shoots up, and it brakes. Switch this one connection off, and the lander never lands at all.

## read.3

Next, the tilt angle feeds the left engine. This is the connection that keeps the lander upright. Without it, only fifteen percent of these flights land.

## read.4

Third, the left leg sensor holds back the right engine. Once a leg is down, the lander stops pushing itself sideways. Without it, the landing rate halves.

## ablation

Here are the connections that matter most, ranked by how much the landing rate drops without them.

## ablation.2

Nineteen of the twenty-six connections can go with almost no loss. A handful of them carry the whole skill.

## ablation.3

And the hidden node? Cut it out completely, and on a hundred test flights the lander does just as well. Seventy-five percent landings without it, seventy-two with it. Evolution added it, but the skill doesn't depend on it. The winner is, in effect, a network with no hidden layer at all.

## harder

So, is the wind hard enough? Here is the champion from the calm-air version of this project, flying in wind, next to the new winner, on the same start.

## harder.2

In calm air, the old champion landed ninety-five percent of the time, with a score of two hundred and fifty-four. In wind, it drops to sixty-four percent, and a hundred and sixteen. The new winner, trained in wind, does better: seventy-two percent, a score of a hundred and sixty-six, and far more landings in the middle. The wind is clearly hard. It is too hard to solve within five hundred generations. But it still didn't force a hidden layer.

## outro

On a hundred new windy flights, the winner lands seventy-two percent of the time, thirty-six percent in the middle, with an average score of a hundred and sixty-six. And it does that with one hidden node it doesn't really need, and twenty-six connections.

## outro.2

The code, the training run, and these videos are all on GitHub. Clone it, train your own, and watch your network grow.
