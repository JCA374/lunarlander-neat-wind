"""MemoryGenome: neat-python's DefaultGenome, plus protected memory nodes when num_hidden > 0.

With num_hidden = 0 (runs A, B, C) it behaves exactly like DefaultGenome. With num_hidden = N:

- Every genome starts with the same N hidden nodes, with the same keys (num_outputs .. num_outputs+N-1).
  Stock neat-python gives each genome's starting hidden nodes different keys, so crossover could never
  line them up.
- Each memory node gets every input, feeds every output, and has a loop to itself that starts at weight
  1.0: a running total (an integrator) from generation 0. Evolution tunes all of it from there.
- Memory nodes are never deleted, and their loops are never deleted or disabled. Everything else
  evolves as usual.

FrozenMemoryGenome (run E) also keeps every memory loop at exactly LOOP_WEIGHT, so each memory node can only
keep a running total of what it is fed. Evolution chooses what to feed it and how to use it.
"""

from itertools import count
from random import choice

import neat

LOOP_WEIGHT = 1.0


def memory_keys(config):
    """Keys of the protected memory nodes (gc = config.genome_config)."""
    n0 = len(config.output_keys)
    return tuple(range(n0, n0 + config.num_hidden))


class MemoryGenome(neat.DefaultGenome):
    def configure_new(self, config):
        n = config.num_hidden
        config.num_hidden = 0
        try:
            super().configure_new(config)      # outputs, input -> output, output loops if recurrent
        finally:
            config.num_hidden = n
        keys = memory_keys(config)
        for k in keys:
            self.nodes[k] = self.create_node(config, k)
            for a, b in [(i, k) for i in config.input_keys] + [(k, o) for o in config.output_keys]:
                self.connections[(a, b)] = self.create_connection(config, a, b)
            loop = self.create_connection(config, k, k)
            loop.weight = LOOP_WEIGHT
            self.connections[(k, k)] = loop
        if keys and config.node_indexer is None:
            config.node_indexer = count(max(keys) + 1)   # new hidden nodes never reuse these keys

    def mutate(self, config):
        self._memory = memory_keys(config)
        super().mutate(config)
        for k in self._memory:                   # the memory loops stay switched on
            if (k, k) in self.connections:
                self.connections[(k, k)].enabled = True

    def mutate_delete_node(self, config):
        protected = set(config.output_keys) | set(memory_keys(config))
        available = [k for k in self.nodes if k not in protected]
        if not available:
            return -1
        del_key = choice(available)
        for key in [k for k in self.connections if del_key in k]:
            del self.connections[key]
        del self.nodes[del_key]
        return del_key

    def mutate_delete_connection(self):
        protected = {(k, k) for k in getattr(self, '_memory', ())}
        keys = [k for k in self.connections if k not in protected]
        if keys:
            del self.connections[choice(keys)]


class FrozenMemoryGenome(MemoryGenome):
    def mutate(self, config):
        super().mutate(config)
        for k in self._memory:
            if (k, k) in self.connections:
                self.connections[(k, k)].weight = LOOP_WEIGHT
