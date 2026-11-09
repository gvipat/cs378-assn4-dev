# Trace format (`agentkv-trace/1`)

A trace fixes all of the token work: every prompt, every output, and every
tool delay. A policy changes only when calls run, where, and which prefixes
stay cached. You (and your agents) can read the public traces;
`agentkv.trace.load(path)` parses one into jobs with token counts and block
hashes.

```json
{
  "schema": "agentkv-trace/1",
  "trace_id": "dev-D-0",
  "family": "D",
  "split": "public_dev",
  "description": "free text for humans",
  "tokenizer": "qwen3-0.6b@c1899de2",
  "memory": { "num_blocks": 192, "block_size": 256 },
  "engine": { "max_num_seqs": 16, "max_num_batched_tokens": 8192 },
  "generation": {
    "generator": "tracegen.dfam",
    "seed": 0,
    "sources": ["data/corpus (English Wikipedia, CC BY-SA 4.0)"]
  },
  "segments": {
    "s3f2a...": {
      "text": "<|im_start|>system\n...<|im_end|>\n",
      "n_tokens": 812
    }
  },
  "jobs": [
    {
      "id": "J00",
      "arrival": 0.0,
      "meta": {},
      "nodes": [
        {
          "id": "n0",
          "type": "llm",
          "stage": "plan",
          "deps": [],
          "prompt": ["s3f2a...", "s91c0..."],
          "output": "s77d1..."
        },
        {
          "id": "n1",
          "type": "tool",
          "stage": "search",
          "deps": ["n0"],
          "delay": 0.62
        }
      ]
    }
  ]
}
```

- `segments` maps a segment id to its text and token count. A prompt is the
  concatenation of its segments, each tokenized separately. Repeated leading
  segments produce shared prefixes; cache matching uses full token blocks,
  not segment ids.
- `type: "llm"` nodes have a `prompt` (a list of segment ids) and an `output`
  (one segment id, whose tokens are the frozen output). `type: "tool"` nodes
  have a `delay` in seconds; think time is a tool node with stage `think`.
- `deps` name nodes in the same job. A job finishes when all its nodes finish;
  its latency is that time minus its `arrival`.
- A policy sees `stage` only as an opaque id, and never sees `family`, `split`,
  `trace_id`, `description`, `generation`, or `meta`.
- Chat turns use Qwen3 control tokens, and each turn's prompt is the previous
  prompt plus its output plus the new messages, so history is shared as a prefix.

W1 comes from the Mooncake conversation trace and W5 from the Qwen Bailian
thinking trace, which publish token counts and block hashes but no text. Their
segments are placeholder words chosen so that prompts sharing a block in the
source share it here, with exact token counts.

## Simulator settings

Trace files use 256-token blocks. The simulator rebuilds prefix hashes in
16-token blocks with the same KV token quota per replica. It replaces the
trace's engine settings with a scheduler limit of 16 calls per replica and a
default batch budget of 2,048 tokens. `target_concurrency` limits total calls
in flight across replicas to 1–16; `batch_token_budget` can choose 512, 1,024,
2,048, or 4,096 tokens. Arrival times, dependencies, tool delays, and prompt and
output lengths stay unchanged. Some prompts
reach 32K tokens, beyond Llama-3-8B's 8K context; the simulator uses them for
timing only. See [The simulator](../README.md#the-simulator).
