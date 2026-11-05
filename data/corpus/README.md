# Part 1 corpus

48 English Wikipedia articles on six systems topics, each frozen at two
revisions. `docs.json` holds cleaned plain text (references, templates, tables
and media removed; each snapshot capped at about 7,000 characters at a
paragraph boundary) plus the SHA-256 of each snapshot's text, which the
citation checker uses as the snapshot hash.

- **v1**: last revision before 2023-01-01T00:00:00Z
- **v2**: last revision before 2025-07-01T00:00:00Z
- v1 stepped back to before 2021-01-01T00:00:00Z when texts are identical

Notes from the build:

- page-replacement: v1 stepped back to before 2021-01-01
- bsp: v1 stepped back to before 2021-01-01
- work-stealing: v1 stepped back to before 2021-01-01

## License and attribution

The text is from Wikipedia, the free encyclopedia, and is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/).
It has been modified (wiki markup stripped and truncated). The authors of
each revision are listed in its history; the permanent links below identify
the exact revisions used. Redistributions of this corpus must keep this
notice and the same license.

| doc_id                      | topic        | article                                                | v1 revision                                                         | v2 revision                                                         |
| --------------------------- | ------------ | ------------------------------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `paxos`                     | consensus    | Paxos (computer science)                               | [1130562313](https://en.wikipedia.org/w/index.php?oldid=1130562313) | [1298087815](https://en.wikipedia.org/w/index.php?oldid=1298087815) |
| `raft`                      | consensus    | Raft (algorithm)                                       | [1129200755](https://en.wikipedia.org/w/index.php?oldid=1129200755) | [1293037883](https://en.wikipedia.org/w/index.php?oldid=1293037883) |
| `two-phase-commit`          | consensus    | Two-phase commit protocol                              | [1125498934](https://en.wikipedia.org/w/index.php?oldid=1125498934) | [1293489049](https://en.wikipedia.org/w/index.php?oldid=1293489049) |
| `three-phase-commit`        | consensus    | Three-phase commit protocol                            | [1032236357](https://en.wikipedia.org/w/index.php?oldid=1032236357) | [1263122209](https://en.wikipedia.org/w/index.php?oldid=1263122209) |
| `byzantine-fault`           | consensus    | Byzantine fault                                        | [1125034931](https://en.wikipedia.org/w/index.php?oldid=1125034931) | [1277093471](https://en.wikipedia.org/w/index.php?oldid=1277093471) |
| `state-machine-replication` | consensus    | State machine replication                              | [1115458341](https://en.wikipedia.org/w/index.php?oldid=1115458341) | [1292141235](https://en.wikipedia.org/w/index.php?oldid=1292141235) |
| `zookeeper`                 | consensus    | Apache ZooKeeper                                       | [1129120707](https://en.wikipedia.org/w/index.php?oldid=1129120707) | [1291004638](https://en.wikipedia.org/w/index.php?oldid=1291004638) |
| `consensus`                 | consensus    | Consensus (computer science)                           | [1116217697](https://en.wikipedia.org/w/index.php?oldid=1116217697) | [1296427069](https://en.wikipedia.org/w/index.php?oldid=1296427069) |
| `lsm-tree`                  | storage      | Log-structured merge-tree                              | [1120239072](https://en.wikipedia.org/w/index.php?oldid=1120239072) | [1268544149](https://en.wikipedia.org/w/index.php?oldid=1268544149) |
| `b-tree`                    | storage      | B-tree                                                 | [1130099925](https://en.wikipedia.org/w/index.php?oldid=1130099925) | [1296531780](https://en.wikipedia.org/w/index.php?oldid=1296531780) |
| `write-ahead-logging`       | storage      | Write-ahead logging                                    | [1106977525](https://en.wikipedia.org/w/index.php?oldid=1106977525) | [1296992876](https://en.wikipedia.org/w/index.php?oldid=1296992876) |
| `rocksdb`                   | storage      | RocksDB                                                | [1130115219](https://en.wikipedia.org/w/index.php?oldid=1130115219) | [1296604118](https://en.wikipedia.org/w/index.php?oldid=1296604118) |
| `leveldb`                   | storage      | LevelDB                                                | [1108873612](https://en.wikipedia.org/w/index.php?oldid=1108873612) | [1296643188](https://en.wikipedia.org/w/index.php?oldid=1296643188) |
| `bloom-filter`              | storage      | Bloom filter                                           | [1130205289](https://en.wikipedia.org/w/index.php?oldid=1130205289) | [1298038948](https://en.wikipedia.org/w/index.php?oldid=1298038948) |
| `log-structured-fs`         | storage      | Log-structured file system                             | [1086540435](https://en.wikipedia.org/w/index.php?oldid=1086540435) | [1286271538](https://en.wikipedia.org/w/index.php?oldid=1286271538) |
| `copy-on-write`             | storage      | Copy-on-write                                          | [1129515604](https://en.wikipedia.org/w/index.php?oldid=1129515604) | [1290859856](https://en.wikipedia.org/w/index.php?oldid=1290859856) |
| `cache-replacement`         | caching      | Cache replacement policies                             | [1114056358](https://en.wikipedia.org/w/index.php?oldid=1114056358) | [1294296010](https://en.wikipedia.org/w/index.php?oldid=1294296010) |
| `page-replacement`          | caching      | Page replacement algorithm                             | [990058626](https://en.wikipedia.org/w/index.php?oldid=990058626)   | [1286567476](https://en.wikipedia.org/w/index.php?oldid=1286567476) |
| `cpu-cache`                 | caching      | CPU cache                                              | [1126673140](https://en.wikipedia.org/w/index.php?oldid=1126673140) | [1297241831](https://en.wikipedia.org/w/index.php?oldid=1297241831) |
| `tlb`                       | caching      | Translation lookaside buffer                           | [1128437264](https://en.wikipedia.org/w/index.php?oldid=1128437264) | [1298112775](https://en.wikipedia.org/w/index.php?oldid=1298112775) |
| `cache-coherence`           | caching      | Cache coherence                                        | [1078177075](https://en.wikipedia.org/w/index.php?oldid=1078177075) | [1292490736](https://en.wikipedia.org/w/index.php?oldid=1292490736) |
| `memory-hierarchy`          | caching      | Memory hierarchy                                       | [1086513241](https://en.wikipedia.org/w/index.php?oldid=1086513241) | [1279503331](https://en.wikipedia.org/w/index.php?oldid=1279503331) |
| `page-cache`                | caching      | Page cache                                             | [1127477511](https://en.wikipedia.org/w/index.php?oldid=1127477511) | [1278568816](https://en.wikipedia.org/w/index.php?oldid=1278568816) |
| `belady-anomaly`            | caching      | Bélády's anomaly                                       | [1079327437](https://en.wikipedia.org/w/index.php?oldid=1079327437) | [1295629789](https://en.wikipedia.org/w/index.php?oldid=1295629789) |
| `tcp-congestion`            | networking   | TCP congestion control                                 | [1125246778](https://en.wikipedia.org/w/index.php?oldid=1125246778) | [1296433353](https://en.wikipedia.org/w/index.php?oldid=1296433353) |
| `bufferbloat`               | networking   | Bufferbloat                                            | [1121151540](https://en.wikipedia.org/w/index.php?oldid=1121151540) | [1292250296](https://en.wikipedia.org/w/index.php?oldid=1292250296) |
| `consistent-hashing`        | networking   | Consistent hashing                                     | [1119161613](https://en.wikipedia.org/w/index.php?oldid=1119161613) | [1292260930](https://en.wikipedia.org/w/index.php?oldid=1292260930) |
| `cdn`                       | networking   | Content delivery network                               | [1128110996](https://en.wikipedia.org/w/index.php?oldid=1128110996) | [1296047956](https://en.wikipedia.org/w/index.php?oldid=1296047956) |
| `rdma`                      | networking   | Remote direct memory access                            | [1116989542](https://en.wikipedia.org/w/index.php?oldid=1116989542) | [1295082598](https://en.wikipedia.org/w/index.php?oldid=1295082598) |
| `quic`                      | networking   | QUIC                                                   | [1122795660](https://en.wikipedia.org/w/index.php?oldid=1122795660) | [1294855284](https://en.wikipedia.org/w/index.php?oldid=1294855284) |
| `load-balancing`            | networking   | Load balancing (computing)                             | [1119414906](https://en.wikipedia.org/w/index.php?oldid=1119414906) | [1296319675](https://en.wikipedia.org/w/index.php?oldid=1296319675) |
| `head-of-line`              | networking   | Head-of-line blocking                                  | [1125765771](https://en.wikipedia.org/w/index.php?oldid=1125765771) | [1256736177](https://en.wikipedia.org/w/index.php?oldid=1256736177) |
| `gpu`                       | accelerators | Graphics processing unit                               | [1129233369](https://en.wikipedia.org/w/index.php?oldid=1129233369) | [1296886937](https://en.wikipedia.org/w/index.php?oldid=1296886937) |
| `cuda`                      | accelerators | CUDA                                                   | [1130323182](https://en.wikipedia.org/w/index.php?oldid=1130323182) | [1298142890](https://en.wikipedia.org/w/index.php?oldid=1298142890) |
| `tpu`                       | accelerators | Tensor Processing Unit                                 | [1127839883](https://en.wikipedia.org/w/index.php?oldid=1127839883) | [1296365375](https://en.wikipedia.org/w/index.php?oldid=1296365375) |
| `hbm`                       | accelerators | High Bandwidth Memory                                  | [1129245322](https://en.wikipedia.org/w/index.php?oldid=1129245322) | [1296618133](https://en.wikipedia.org/w/index.php?oldid=1296618133) |
| `systolic-array`            | accelerators | Systolic array                                         | [1099212479](https://en.wikipedia.org/w/index.php?oldid=1099212479) | [1296425722](https://en.wikipedia.org/w/index.php?oldid=1296425722) |
| `gpgpu`                     | accelerators | General-purpose computing on graphics processing units | [1128345183](https://en.wikipedia.org/w/index.php?oldid=1128345183) | [1296423261](https://en.wikipedia.org/w/index.php?oldid=1296423261) |
| `simt`                      | accelerators | Single instruction, multiple threads                   | [1123650169](https://en.wikipedia.org/w/index.php?oldid=1123650169) | [1293920904](https://en.wikipedia.org/w/index.php?oldid=1293920904) |
| `roofline`                  | accelerators | Roofline model                                         | [1091143018](https://en.wikipedia.org/w/index.php?oldid=1091143018) | [1280397259](https://en.wikipedia.org/w/index.php?oldid=1280397259) |
| `mapreduce`                 | dataparallel | MapReduce                                              | [1126559707](https://en.wikipedia.org/w/index.php?oldid=1126559707) | [1262698358](https://en.wikipedia.org/w/index.php?oldid=1262698358) |
| `spark`                     | dataparallel | Apache Spark                                           | [1119600867](https://en.wikipedia.org/w/index.php?oldid=1119600867) | [1294860679](https://en.wikipedia.org/w/index.php?oldid=1294860679) |
| `hadoop`                    | dataparallel | Apache Hadoop                                          | [1130280585](https://en.wikipedia.org/w/index.php?oldid=1130280585) | [1297299791](https://en.wikipedia.org/w/index.php?oldid=1297299791) |
| `bsp`                       | dataparallel | Bulk synchronous parallel                              | [989641291](https://en.wikipedia.org/w/index.php?oldid=989641291)   | [1292515773](https://en.wikipedia.org/w/index.php?oldid=1292515773) |
| `work-stealing`             | dataparallel | Work stealing                                          | [990787763](https://en.wikipedia.org/w/index.php?oldid=990787763)   | [1292171399](https://en.wikipedia.org/w/index.php?oldid=1292171399) |
| `amdahl`                    | dataparallel | Amdahl's law                                           | [1123971320](https://en.wikipedia.org/w/index.php?oldid=1123971320) | [1296424385](https://en.wikipedia.org/w/index.php?oldid=1296424385) |
| `gustafson`                 | dataparallel | Gustafson's law                                        | [1107600217](https://en.wikipedia.org/w/index.php?oldid=1107600217) | [1285948771](https://en.wikipedia.org/w/index.php?oldid=1285948771) |
| `dataflow`                  | dataparallel | Dataflow programming                                   | [1123188424](https://en.wikipedia.org/w/index.php?oldid=1123188424) | [1286656491](https://en.wikipedia.org/w/index.php?oldid=1286656491) |

## Regenerating

Staff rebuild the corpus from the pinned revisions above. The pins are
date-based, so a rebuild selects the same revisions unless a revision is
deleted or suppressed on Wikipedia; `--check` reports any hash drift.
