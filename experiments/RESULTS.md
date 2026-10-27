# STC re-examination: experiments

Backend for every BWT number below: `bsc 3.3.5 -b100p -e2` (enwik8) / `-b1000p -e2` (enwik9).
The paper's own M03 coder was not built (the repo's code was not run). Side-model numbers are
ideal code lengths (sum of -log2 p) from causal models that only use decoder-side information.

## Reproduction of the STC split

`stc_split.py` reimplements Sec. 3.4 / Table 1 of the paper from the text. On enwik9 its 13 side
streams have exactly the paper's raw sizes (Table 4), 20,050,492 bytes in total.

| enwik9 | paper (M03) | here (bsc) |
|---|---:|---:|
| no split | 160,017,749 | 163,924,696 |
| masked main stream | 148,105,611 | 151,888,944 |
| STC side streams | 9,282,534 | 9,396,368 |
| STC gain vs no split | -2,629,561 | -2,639,384 |

## Digit payload: what a principled conditional model gets

Same skeleton (digits -> '9'), digits coded in document order by `dcm.c`.

| digit payload coded as | enwik8 bytes | enwik9 bytes |
|---|---:|---:|
| inline in the BWT (bsc(x) - bsc(masked)) | 868,208 | 12,035,752 |
| order-0 entropy | 864,352 | 13,838,321 |
| raw digits in document order, bsc | 739,686 | 11,537,878 |
| STC side streams, bsc | 605,146 | 9,396,368 |
| STC side streams, M03 (paper) | 595,257 | 9,282,534 |
| direct model, same information as STC's keys | 585,427 | 8,768,234 |
| + deeper left + right-side context | 574,450 | |
| + previous value of the same field | 551,964 | |
| all contexts (+ word before, same-article, doc order) | 527,654 | 7,648,542 |

Recency vs. context (enwik8): context only 585,427; recency only 619,245; both mixed 556,273.

Value models on XML header fields (replacing the string model's cost for those runs):

| field | enwik8 string -> value | enwik9 string -> value |
|---|---|---|
| timestamp, predicted from revision id | 37,326 -> 19,977 | 748,083 -> 365,087 |
| page id, delta coded | 2,911 -> 1,497 | 47,096 -> 33,513 |

## Other classes ("more than numbers"), enwik8, digits already masked

| class projected out | BWT saving | side cost | net |
|---|---:|---:|---:|
| digits (dcm all contexts) | 868,208 | 527,654 | -340,554 |
| non-ASCII bytes (`ccm.c u`) | 154,444 | 146,090 | -8,354 |
| UTF-8 continuation bytes (`ccm.c c`) | 123,960 | 122,305 | -1,655 |
| letter case (`casecm.c`) | 206,070 | 291,478 | +85,408 |

`ib.py` (agglomerative Information Bottleneck over bytes, relevance = previous and next byte)
merges UTF-8 bytes first, then digits; at 12 clusters the digits plus '%' form their own cluster.

## Running

```bash
E8=/Users/jordigonzalez/wget/gpusais/datasets/enwik8
python3 -I stc_split.py $E8 out8 && ./bscsize.sh "-b100p -e2" out8/s*.bin
cc -O3 -march=native -o dcm dcm.c -lm && ./dcm $E8 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 255 0.006 percost.bin
python3 -I valuemodels.py $E8 percost.bin
```

For enwik9, build `dcm` with `TBITS 24` and `FBITS 22` (larger hash tables).

## Brown-style abstraction under the right objective (`mdlctx.c`, enwik8)

Order-k context model, exact (unhashed) contexts. `fit` = N·Ĥ(target | abstracted context), the
maximum-likelihood objective Brown clustering optimizes; `kt` = sequential KT code length (fit plus
learning cost), a proper MDL code. Raw output: `mdl_results_enwik8.txt`.

Digits merged in contexts only, full-byte target (change vs. no abstraction):

| order k | fit | MDL code length |
|---:|---:|---:|
| 1 | +49,438 | +47,823 |
| 2 | +123,852 | +71,651 |
| 3 | +231,468 | -90,413 |
| 4 | +357,318 | -500,932 |
| 5 | +470,433 | -905,334 |
| 6 | +517,246 | -1,197,381 |

Per-decision partitions, k = 4 (class = "which non-digit byte, or a digit"; member = "which digit"):

| configuration | bytes |
|---|---:|
| no abstraction | 32,333,814 |
| Brown's form: digits merged, member coded with no context | 31,903,208 |
| one partition for everything (digits merged in both decisions) | 31,768,737 |
| per decision: merged for the class, literal for the member | 31,713,338 |
| + case folded in the class decision's contexts | 31,684,657 |

Context-only abstractions ranked (full-byte target):

| abstraction | fit, k=4 | MDL, k=4 | MDL, k=5 |
|---|---:|---:|---:|
| digits -> one | +357,318 | -500,932 | -905,334 |
| case fold | +1,032,720 | -44,716 | -594,840 |
| .,;:!? -> one | +272,984 | -37,078 | -161,630 |
| bytes >= 0x80 -> one | +335,812 | -12,738 | -86,180 |
| newline/tab -> space | +313,727 | +139,372 | +19,500 |
| capitals -> one | +1,908,387 | +364,004 | -252,211 |
| vowels -> one | +6,299,203 | +4,037,663 | +1,589,168 |

## Automatic MDL-scored merge search (`mdlbrown.c`, enwik8)

Each byte = 8 binary decisions, MSB first (node 0 = bit 7, ..., node 7 = bit 0), each predicted from
an order-4 context whose symbols pass through that node's own partition. Merges are proposed by
Information Bottleneck similarity (top 60 pairs) and accepted only if the exact KT code length drops.
Full logs per node: `mdlbrown_enwik8/`.

| run | start | final | change | fit change |
|---|---:|---:|---:|---:|
| separate partition per node (sum of 8 runs) | 27,788,195 | 27,363,779 | -424,416 | +313,906 |
| one shared partition for all 8 nodes | 27,788,196 | 27,589,751 | -198,445 | +220,355 |

Where the digits ended up:

| node | what it decides (ASCII) | digit classes found |
|---|---|---|
| 0 | ASCII vs. non-ASCII | merged with letters/punctuation (18 classes total) |
| 1 | 0x40-0x7F vs. below | {34567} {28} {9$}; 0, 1 separate |
| 2 | case-ish bit (0x20 block) | {$0123456789}: all ten digits |
| 3-6 | low bits | {345678} or {3456}{78}; 0, 1, 2, 9 kept separate |
| 7 | last bit | {345678} {19} |
| shared | all of the above | {345678} only |

## Next level: soft sharing, mixing, and a learned model (enwik8)

`lattice.c`: bitwise order-6 model. Each context node shrinks toward a learned blend of two parents,
its shorter suffix and its more abstract version (digit-masked, then also case-folded). Optional
logistic mixer over all node predictions. A node only uses its abstraction parent when the abstraction
changes its window (otherwise the parent duplicates the node). Logs: `next_level_logs/`.

16 MB slice (`enwik8_16m`), order 6, bytes:

| model | no mixer | mixer |
|---|---:|---:|
| literal contexts only | 4,277,521 | 3,978,782 |
| hard digit merge | 4,246,922 | 3,957,785 |
| hard digit + case merge | 4,352,873 | 4,049,251 |
| lattice: literal + digit parents | 4,247,968 | 3,919,733 |
| lattice: literal + digit + case parents | 4,264,014 | 3,907,481 |
| online LSTM (`neural.py`, 2x512, nothing pre-trained) | 4,090,369 | |

Full enwik8, order 6, mixer on:

| model | total | digit bytes | other bytes |
|---|---:|---:|---:|
| literal contexts only | 22,785,397 | 908,092 | 21,877,305 |
| hard digit merge | 22,702,506 | 958,099 | 21,744,407 |
| lattice (literal + digit + case) | 22,600,998 | 893,220 | 21,707,778 |

What the LSTM learned unprompted (`inspect_emb.py`, `next_level_logs/lstm_embeddings.txt`):
output embeddings cluster all ten digits, all capitals and all lowercase letters; input embeddings
pull case pairs together (cos +0.22 vs -0.02 for mismatched letters) and cluster 1-9 but put 0 with
`% . /` (round numbers and decimals).

## Same comparison on a strong base (`cmlat*.c`, full enwik8)

`cmlat.c`: context mixing with orders 1-7, two word contexts and a match model per abstraction
level, adaptive counters, two mixers, final APM; ideal code length. `cmlat_d.c` skips abstracted
contexts identical to the finer level (and takes `-DTB=` for table size); `cmlat_g.c` with `GATE=1`
only lets literal contexts vote once the coded high nibble says "digit block".

| model | 512 MB table | 1 GB table |
|---|---:|---:|
| literal | 20,768,114 | 20,495,184 |
| hard digit merge | 20,584,334 | 20,334,826 |
| hard digit + case merge | 20,886,745 | 20,660,795 |
| lattice literal + digit, naive | 20,706,635 | |
| lattice literal + digit, deduplicated | 20,566,039 | 20,301,276 |
| lattice literal + digit + case, deduplicated | | 20,305,460 |
| lattice, literal contexts gated to digits | | 20,306,979 |

Reference points: bsc on LTCB 20,293,393; local bsc 3.3.5 best 20,786,794; STC 19,996,029.

Best measured configuration against STC (STC's M03 skeleton from its report; side parts are ideal
code lengths from `dcm.c` and `valuemodels.py`):

| configuration | bytes |
|---|---:|
| STC as published | 19,996,029 |
| STC skeleton + `dcm` digit model instead of STC's side streams | 19,928,426 |
| + timestamp-from-revision-id and page-id delta models | 19,909,663 |

## A coder that beats M03 like-for-like (`cm2.c`, full enwik8)

Adds what the semi-static suffix-tree coder has over `cmlat.c`: statistic sharing (bit-history states
mapped to probabilities by a StateMap shared across contexts of one type, alongside direct
estimates), long orders (1-6, 8, 12, 16, 24), 2-way hash buckets with checksums. 512 MB table.
Ideal code length; not yet an arithmetic-coded round trip.

| coder | enwik8 bytes |
|---|---:|
| bsc (LTCB) | 20,293,393 |
| bsc-m03 (M03, no tricks) | 20,263,925 |
| STC (M03 + digit split) | 19,996,028 |
| `cm2` literal contexts only (no tricks) | 19,457,552 |
| `cm2` hard digit merge | 19,297,195 |
| `cm2` lattice: literal + digit-masked, deduplicated | **19,266,052** |
| `cm2` lattice + case-folded | 19,276,371 |

Digit bytes: literal 776,516, hard merge 802,281, lattice 748,011.

## enwik9 (`cm2p.c` built with `-DTB=29`, a 2 GB table; ideal code length, no decoder counted)

| Coder | enwik9 bytes | vs M03 |
|---|---|---|
| `bsc-m03` 0.5.5 (local) | 160,364,392 | — |
| STC's same-coder no-split control | 160,017,749 | −0.2% |
| **`cm2` literal (no tricks)** | **157,582,526** | **−1.7%** |
| STC (archive) | 157,388,188 | −1.9% |
| STC (archive + decoder) | 157,571,362 | −1.7% |
| **`cm2` lattice 01 (digits→'9' as an extra context view)** | **155,305,085** | **−3.2%** |

Digit bytes: literal 10,006,643 → lattice 9,648,838 (−3.6%). Non-digit bytes: 147,575,884 → 145,656,247 (−1.3%).
Masked contexts also help the text, because they let templates with differing numbers share statistics.
The literal coder alone roughly matches all of STC: 194 KB behind the archive, and level with archive + decoder.
The lattice beats STC by 2.08 MB.
Caveat: these are ideal code lengths only, not a round-tripped encoder and decoder.

## Pushing further: `cm3`–`cm7` (feature-flagged A/B tests on enwik8_16m, then full enwik8)

Each feature was tested alone on 16 MB with levels 01. Logs are in `ab16m_logs/`. Bytes saved are relative to the build without that feature:

| Feature (bit) | Bytes saved on 16 MB |
|---|---|
| mixer weight set chosen by the previous byte (16) | 24.4K |
| mixer weight set chosen by word length × byte class × match bucket (128) | 15.3K |
| case-folded word contexts (32768) | 15.3K |
| indirect contexts: bytes that followed this order-1/order-2 context last time (256) | 14.9K |
| text contexts: sparse, column, markup, three-word (4) | 12.0K |
| run input per context (64) | 11.6K |
| second, slow direct probability per slot (262144) | 22K |
| 8-way 64-byte buckets with 16-bit checks, evict least-visited (1048576) | 17K at 2^27 slots, 7K at 2^29 |
| larger table: 2^29 slots instead of 2^27 | 34K |
| mixer weight set chosen by the byte two back; lower learning-rate floor; order-3 APM; sparse 2; learned final mixer | 2–6K each |
| no gain: match-state APM, learned APM blend, higher learning rate, gentler history discount, chained SSE, orders 7/10/32, StateMap/APM rates | ≤1K or worse |

16 MB totals: the cm2-equivalent build came to 3,416,592 bytes, and the best `cm7` build to 3,211,817 (−6.0%).

Full enwik8 run: `cm7` built with `-DTB=29 -DORDSET=1 -DSMLIM=255`, features 1499647. These are ideal code lengths.

| Coder | enwik8 bytes |
|---|---|
| `bsc-m03` | 20,263,925 |
| STC | 19,996,028 |
| `cm2` lattice 01 (previous best) | 19,266,052 |
| **`cm7` literal (level 0)** | **17,824,686** |
| **`cm7` lattice 01** | **17,710,593** |

On full enwik8, `cm7` lattice 01 is 8.1% smaller than `cm2` and 11.4% smaller than STC. The literal coder alone is 12.0% smaller than M03. The digit lattice is still worth 114 KB on top of the literal coder.

### `cm7` on enwik9 (same build as the enwik8 run: 2^29 slots, 4 GB table; ideal code length, no decoder counted)

| Coder | enwik9 bytes | vs STC |
|---|---|---|
| `bsc-m03` 0.5.5 | 160,364,392 | +1.9% |
| STC (archive) | 157,388,188 | — |
| `cm2` literal | 157,582,526 | +0.1% |
| `cm2` lattice 01 | 155,305,085 | −1.3% |
| **`cm7` literal (level 0)** | **144,524,953** | **−8.2%** |
| **`cm7` lattice 01** | **143,237,069** | **−9.0%** |

The `cm7` literal coder is 9.9% smaller than M03 like-for-like. The digit lattice adds a further 1.29 MB. Digit bytes go from 9,340,032 to 9,100,490 (−2.6%) and the rest from 135,184,925 to 134,136,583 (−0.8%).
Both runs used 2^29 slots, which is probably table-limited at 1 GB of input. On the 16 MB slice, 2^27 → 2^29 slots was still worth 0.7–1%.
