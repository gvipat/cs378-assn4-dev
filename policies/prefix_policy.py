"""Prefix-aware eviction and cache-aware routing (Part 2, tasks 2.2 and 2.3).

choose_victim: evict the least recently used cached block that is a leaf of
its replica's prefix tree (no resident block on that replica extends it), and
fall back to plain LRU when no candidate is a leaf. Track the tree from alloc
and evict events; do not read view.blocks.

route_call: among replicas whose in_flight is at most `balance_threshold`
above the least loaded replica's, send the call to the one holding the most
cached blocks of its prompt; break ties by lower in_flight, then lower replica
id. This follows the cache-aware policy of SGLang's router.

params: cap (default 8); leaf_first (default True; False gives plain LRU);
balance_threshold (default 1; -1 routes to the least loaded replica, like the
reference). With leaf_first=False and balance_threshold=-1, this policy makes
the same decisions as the reference.

    # Task 2.2 (eviction only):
    pytest tests/test_part2_prefix.py -k "not route"
    # After task 2.3 (eviction and routing):
    pytest tests/test_part2_prefix.py
"""

from collections import OrderedDict

from agentkv.policy import Policy


class PrefixPolicy(Policy):

    def __init__(self, params=None):
        super().__init__(params)
        self.cap = int(self.params.get("cap", 8))
        self.leaf_first = bool(self.params.get("leaf_first", True))
        self.balance_threshold = int(self.params.get("balance_threshold", 1))
        # [Part 2.2] TODO: Set up the state for recency and for each replica's
        # prefix tree (which resident blocks extend which block hash).
        raise NotImplementedError("Part 2: fill PrefixPolicy.__init__")
        # END OF YOUR CODE

    def on_event(self, event, state):
        """Update recency and the prefix trees."""
        # [Part 2.2] TODO: alloc and hit make a block most recent; alloc adds it to
        # its replica's tree under event.parent_hash (None for the first block of
        # a prompt); evict removes it.
        raise NotImplementedError("Part 2: fill PrefixPolicy.on_event")
        # END OF YOUR CODE

    def target_concurrency(self, view, state):
        return self.cap

    def choose_victim(self, view, candidates, state):
        """Return the least recently used leaf among candidates, or the least recently used candidate."""
        # [Part 2.2] TODO: With leaf_first=False, behave exactly like LRU.
        raise NotImplementedError("Part 2: fill PrefixPolicy.choose_victim")
        # END OF YOUR CODE

    def route_call(self, view, call, state):
        """Choose a replica for a ready call."""
        # Supplied for the Part 2.2 experiments: keep least-loaded routing
        # when affinity is disabled or there is only one replica.
        if self.balance_threshold == -1 or len(view.replicas) == 1:
            return super().route_call(view, call, state)
        # [Part 2.3] TODO: Use view.replicas[r].in_flight for load and
        # view.num_cached_blocks(call, r) for the cached prefix.
        raise NotImplementedError("Part 2: fill PrefixPolicy.route_call")
        # END OF YOUR CODE
