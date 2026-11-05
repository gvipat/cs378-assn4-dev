# Data sources and licenses

| Data                                               | Source                                                                                                          | License      |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------ |
| Part 1 corpus (`data/corpus/`) and D deep research | English Wikipedia, 48 articles, 2 pinned revisions each; D traces are runs of the staff agent on this corpus    | CC BY-SA 4.0 |
| W1 long-context chat                               | kvcache-ai/Mooncake conversation trace (FAST 2025): token counts and prefix hashes, with placeholder text       | Apache-2.0   |
| W2 coding agent                                    | sammshen/lmcache-agentic-traces (SWE-bench sessions)                                                            | CC BY 4.0    |
| W3 document QA                                     | bigai-nlco/LooGLE short-dependency QA (ACL 2024)                                                                | CC BY-SA 4.0 |
| W4 customer-service agents                         | sierra-research/tau2-bench published conversations                                                              | MIT          |
| W5 reasoning chat                                  | alibaba-edu/qwen-bailian-usagetraces-anon thinking trace: token counts and prefix hashes, with placeholder text | Apache-2.0   |
| M1, M2 mixes                                       | W3 with W1, and W3 with W5, on one engine                                                                       | As above     |
| Bursty arrivals                                    | kvcache-ai/Mooncake tool-agent trace (timestamps only)                                                          | Apache-2.0   |
| Tokenizer                                          | Qwen/Qwen3-0.6B tokenizer (shared by all Qwen3 sizes)                                                           | Apache-2.0   |
| Simulator                                          | microsoft/vidur @ 25e0082dbbfb                                                                                  | MIT          |

Each trace in `data/traces/` records in its `generation` field the generator,
seed, and settings that produced it, and the exact source datasets and revisions.
The format is in [docs/trace_format.md](../docs/trace_format.md).
