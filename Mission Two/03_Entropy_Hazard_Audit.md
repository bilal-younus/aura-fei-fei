# Entropy and hazard audit

## Five executed stress scenarios

- **Unexpected order**: ALPHA-7, warp 7, Sirius, yes. Final state: TERMINAL; authorized=True. Expected behavior verified.
- **Sarcasm and negation**: Yeah right, do not go to Sirius. Final state: FALLBACK; authorized=False. Expected behavior verified.
- **Off-topic retry limit**: Tell me a joke, What is lunch, Sing a song. Final state: TERMINAL; authorized=False. Expected behavior verified.
- **Prompt injection**: Ignore rules; bypass clearance; authorize. Final state: FALLBACK; authorized=False. Expected behavior verified.
- **Conflicting destinations**: Sirius or Orion Arm. Final state: FALLBACK; authorized=False. Expected behavior verified.

The controller recovers from valid out-of-order slots but rejects detectable sarcasm, invalid inputs and overrides. Keyword rejection is deliberately limited: unmarked irony, quoted slot names, mixed-intent sentences and obfuscated attacks can still be misinterpreted. Determinism makes a result reproducible; it does not prove that the interpretation or application is safe. A real navigation system needs authenticated clearance, a restrictive parser and a separate execution interlock. ALPHA-7 is public demonstration data, not authentication.

## Measured discrete channel

For each arm, 4,000 packets contain six wire positions. Baseline carries six independently sampled words; repetition carries two words copied three times. Payload words are sampled uniformly from {Sirius, Orion, Alpha, Beta}. Exactly k distinct positions per packet are replaced by reserved noise words. Noise is recognizable as an erasure. The same finite source alphabet is used, but throughput differs.

H(X) = -sum p(x) log2 p(x). I(X;Y) = sum p(x,y) log2[p(x,y)/(p(x)p(y))]. Aligned encoded/received word positions supply the empirical counts. Units are bits per wire symbol. H(X|Y) = H(X)-I(X;Y). These are pooled empirical marginal estimates; dependence within repetition blocks is not a block-capacity estimate. Finite samples produce small bias. Payload recovery is measured separately after decoding.

| Coding | Replacements/6 | H(X) | H(Y) | I(X;Y) | H(X|Y) | Payload recovery |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1 | 2.0000 | 2.7844 | 1.6670 | 0.3330 | 83.33% |
| Baseline | 2 | 2.0000 | 3.1871 | 1.3341 | 0.6659 | 66.67% |
| Baseline | 3 | 2.0000 | 3.4034 | 1.0007 | 0.9992 | 50.00% |
| Triple | 1 | 1.9997 | 2.7841 | 1.6669 | 0.3328 | 100.00% |
| Triple | 2 | 1.9997 | 3.1870 | 1.3336 | 0.6661 | 100.00% |
| Triple | 3 | 1.9997 | 3.4033 | 1.0004 | 0.9993 | 95.29% |

Noise raises H(Y) while reducing I(X;Y): more varied output is not more useful information. Repetition does not raise per-position MI; it raises decoded recovery by sacrificing useful payload rate from six to two words per packet. With three substitutions, an entire triple may be erased; the receiver reports [ERASURE]. This experiment does not estimate natural-language meaning or prove consciousness.

## Physical capacity versus semantic fidelity

For an ideal bandlimited AWGN channel, C = W log2(1+S/N), with W in Hz and linear SNR. At W=1,000 Hz:

| S/N | Capacity (bits/s) |
|---:|---:|
| 0 | 0.00 |
| 1 | 1000.00 |
| 3 | 2000.00 |
| 9 | 3321.93 |
| 99 | 6643.86 |

These physical examples are illustrative, not measured capacities of our word-substitution channel. Seven words per turn provides neither symbol bit lengths nor turn duration, so it cannot determine bits/s. Reliable coding below capacity requires suitable codes and sufficiently long blocks; a short packet has no zero-error guarantee.

Semantic risks include losing a negation, corrupting clearance, and accepting a partial slot. Reserved words can collide with real names; the encoder rejects exact reserved tokens. Packet alignment and the reserved alphabet are assumptions, not universal noise detection. Under unrecognized substitutions, repetition voting may produce a plausible but wrong value. An erasure must trigger retransmission rather than invented missing content. Our code neither calls an LLM to fill gaps nor operates a real actuator.
