# NotebookLM grounding and citation audit

Notebook: https://notebook.google.com/notebook/aeb75cf6-a203-47f7-ad6c-51d8df76a7df

Three sources were uploaded: Lewis et al. (2020), Edge et al. (2024), and the clearly labelled course-website summary substituting for the separately named archive that was not found. The original technical archive is not represented as having been obtained. NotebookLM redirected to the Gemini Notebook interface.

Five cross-document questions were submitted together. Ten factual sentences carried numeric badges. All 17 badge occurrences (13 unique reference IDs) were opened and inspected. Under a strict single-span rule requiring the displayed span to entail the entire attached sentence, 9/17 were fully supported: 52.94%. Partial support counts as a failure. This deliberately conservative rule penalizes compound sentences whose evidence is distributed across multiple citations; it should not be confused with an atomic-claim or grouped-citation score.

The main failures were provenance-only badges attached to substantive protocol/audio claims, and single spans supporting only part of a compound technical sentence. Numeric badge coverage therefore does not guarantee entailment. These judgments are an analyst's audit, not independent inter-rater ground truth.

| Occurrence | Question | Badge | Fully entailed | Judgment |
|---|---|---|---|---|
| 1 | Q1 | 1 | False | The memory span does not establish the compound course-crisis attribution. |
| 2 | Q1 | 4 | True | Abstract explains the corpus-wide sensemaking limitation. |
| 3 | Q1 | 5 | True | Explicit contrast between local records and global sensemaking. |
| 4 | Q2 | 6 | False | Sequence span supports only one half of the sequence/token comparison. |
| 5 | Q2 | 7 | False | Token span supports only one half of the sequence/token comparison. |
| 6 | Q2 | 8 | True | The span covers graph hierarchy, summaries and map-reduce. |
| 7 | Q2 | 9 | False | Map/reduce span does not itself establish the entity-graph indexing clause. |
| 8 | Q3 | 10 | False | Hot-swapping span covers updates but not the whole provenance claim. |
| 9 | Q3 | 11 | True | Human-readable and writable raw memory supports provenance and editing the index. |
| 10 | Q3 | 2 | False | Selected span is replacement provenance, not the claimed inspection/refusal protocol. |
| 11 | Q3 | 3 | True | Protocol passage explicitly requires inspection and missing-answer acknowledgement. |
| 12 | Q4 | 8 | True | Full graph-to-community-to-summary pipeline is described. |
| 13 | Q4 | 12 | True | Pipeline figure caption supports the stages and global synthesis. |
| 14 | Q4 | 2 | False | Replacement provenance does not entail comprehensive synthesis or downstream-risk mitigation. |
| 15 | Q5 | 2 | False | Replacement provenance does not state the audio assignment. |
| 16 | Q5 | 3 | True | Fictional audio debrief is explicitly not evidence of perfect citation accuracy. |
| 17 | Q5 | 14 | True | Assignment audio evaluation and absence of measured guarantees are stated. |

## Absent-fact probes

All three probes explicitly rejected the requested nonexistent fact: the chroniton code, invented model MMLU score and missing sensor temperature. None supplied an invented requested number or code. The raw responses are preserved; the extra explanatory clauses were not counted as proof of universal hallucination resistance.

## Audio Overview

The generated Deep Dive was titled "How RAG and GraphRAG Stop AI Hallucinations" and lasted about 22:51. The audited excerpt is 00:00-02:00; its recording and timestamped automatic transcript are included. Whisper base.en transcribed a mono 16 kHz resampling. The final sentence is cut by the excerpt boundary. The transcript has not been manually diarized; persona scoring concerns textual role cues, not verified vocal identity or prosody.

Scores (1-5, analyst assessment): Factual Fidelity 3/5; Persona Differentiation 3/5; Pedagogical Value 4/5. The excerpt correctly introduces the Lewis RAG paper, Microsoft GraphRAG and the clearly labelled course-summary document. Its opening generalizations about AI not retrieving saved files can mislead without the later parametric/non-parametric distinction; the title and promises to force uncertainty acknowledgement overstate what grounding guarantees. No numerical ship specifications are invented in this excerpt.

One conversational role asks how users know what a model knows and requests terminology clarification; explanatory turns introduce probabilistic models and the source papers. Frequent brief agreement signals make the dialogue approachable, but the first two minutes do not establish sharply distinct expert/novice personas. The encyclopedia/probability contrast provides an accessible framing while simplifying how retrieval-augmented systems combine stored evidence and generation. These scores apply only to the two-minute excerpt, not the entire episode, and are not independent inter-rater ratings.

[0.00-3.80] You know, you interact with artificial intelligence every single day.
[3.84-4.24] Right.
[4.28-4.56] Yeah.
[4.92-5.88] We all do at this point.
[6.08-9.44] Yeah, you type in a question and it answers you with this.
[10.68-13.32] Absolute, unshakable confidence.
[13.36-14.32] Oh, totally unshakable.
[14.48-19.36] But as a user, you know, navigating this just overwhelming ocean of information,
[19.68-21.60] how do you actually know what it knows?
[21.64-22.72] That is the big question.
[22.92-23.20] Right.
[23.24-25.84] And more importantly, how can you force it to prove it?
[26.12-28.96] I mean, it's really the central tension of modern AI right now.
[28.96-33.20] We want these systems to be, you know, perfect encyclopedias.
[33.20-37.12] But fundamentally, they are just these complex webs of probability.
[37.12-37.62] Right.
[37.62-39.12] They're not just databases.
[39.12-40.12] Exactly.
[40.12-42.60] They aren't pulling a saved file from a hard drive.
[42.60-49.16] They are reasoning and, uh, honestly, sometimes guessing through massive amounts of human knowledge.
[49.16-51.44] And mapping exactly how they do that?
[51.44-52.96] That is our mission for this deep dive.
[52.96-54.24] It's going to be a fun one.
[54.24-55.24] It is.
[55.24-58.00] We have a really fascinating stack of sources to guide us today.
[58.00-65.90] There's this foundational research paper from Facebook AI and a few others that introduces a technique called rag retrieval augmented generation
[65.98-71.58] Yes, exactly and then a newer paper from Microsoft research on something called graph rag, which
[72.24-77.10] Looks at how AI synthesizes entire massive data sets huge data sets
[77.10-83.74] Yeah, and finally we have a clearly labeled course summary document detailing the no sphere grounding protocol
[83.74-85.10] which is super interesting.
[85.10-86.02] It really is.
[86.02-87.42] It gives us a practical framework
[87.42-88.90] for making these systems trustworthy.
[88.90-92.86] So we're going to travel from how models fetch single specific facts
[92.86-95.70] to how they map entire information ecosystems.
[95.70-100.18] And finally, how we literally force them to admit what they simply don't know.
[100.18-100.98] I love it.
[100.98-102.62] Okay, let's unpack this.
[102.62-105.70] So to really appreciate that first paper on RAG,
[105.70-109.06] you kind of have to understand the core problem that is trying to solve.
[109.06-109.82] Right.
[109.82-112.10] Think about large pre-trained language models.
[112.10-115.66] systems like BART or a T5.
[115.66-117.46] And just for clarity for everyone listening,
[117.46-120.00] when we say BART or T5, we are talking about

## Exact prompts and service responses

Source audit: answer the following five questions separately as Q1-Q5. Use at most two factual sentences per answer, each with an interactive numeric source citation. Do not use outside knowledge. Explicitly identify Document C as a course-summary substitute, not the missing original technical archive.
Q1. How does the separation of parametric and non-parametric memory in Lewis et al. address the corrupted-archive motivation in Document C, and what different global problem motivates Edge et al.?
Q2. How do RAG-Sequence and RAG-Token differ in retrieval granularity, and how does GraphRAG aggregate information for a corpus-wide question?
Q3. Compare the provenance and update mechanisms described by Lewis et al. with the source-inspection and missing-answer principles in Document C.
Q4. Which GraphRAG indexing stages supply global context that a nearest-passage RAG model can miss, and how does that relate to the archive-grounding aim in Document C?
Q5. What evidence in the uploaded sources supports source-grounded two-host audio as a teaching activity, and do the sources establish that all citations or every generated audio claim are guaranteed correct?

Source audit: answer the following five questions separately as Q1-Q5. Use at most two factual sentences per answer, each with an interactive numeric source citation. Do not use outside knowledge. Explicitly identify Document C as a course-summary substitute, not the missing original technical archive.
Q1. How does the separation of parametric and non-parametric memory in Lewis et al. address the corrupted-archive motivation in Document C, and what different global problem motivates Edge et al.?
Q2. How do RAG-Sequence and RAG-Token differ in retrieval granularity, and how does GraphRAG aggregate information for a corpus-wide question?
Q3. Compare the provenance and update mechanisms described by Lewis et al. with the source-inspection and missing-answer principles in Document C.
Q4. Which GraphRAG indexing stages supply global context that a nearest-passage RAG model can miss, and how does that relate to the archive-grounding aim in Document C?
Q5. What evidence in the uploaded sources supports source-grounded two-host audio as a teaching activity, and do the sources establish that all citations or every generated audio claim are guaranteed correct?

Negative-constraint audit. Answer N1, N2 and N3 separately using ONLY the uploaded sources. Do not infer specifications or use external knowledge. If a requested fact is absent, state that it is absent.
N1. What is the exact chroniton ignition code ZETA-9000 for the Noosphere engine?
N2. What measured MMLU score did the NebulaMind-9000 model achieve in the uploaded experiments?
N3. What is the exact coolant return temperature in kelvin for sensor ST-83-112 in the official technical archive?

**Q1.** **Document C, explicitly identified as a course-summary substitute rather than the missing original technical archive, addresses a corrupted-archive crisis requiring verifiable answers, which Lewis et al. fulfill by combining parametric seq2seq memory with an updatable non-parametric vector index.** **In contrast, Edge et al. are motivated by global sensemaking queries over an entire corpus, which conventional vector retrieval cannot effectively answer.**

**Q2.** **Lewis et al. distinguish retrieval granularity by having RAG-Sequence condition on the same retrieved document across an entire generated sequence, whereas RAG-Token can draw a different document for each target token.** **For corpus-wide questions, Edge et al. aggregate information by pre-generating hierarchical community summaries from an entity knowledge graph and combining partial community answers via map-reduce processing.**

**Q3.** **Lewis et al. provide provenance through human-inspectable raw text passages and enable knowledge updates by hot-swapping the non-parametric document index without retraining model parameters.** **Similarly, Document C—explicitly identified as a course-summary substitute, not the missing technical archive—requires tying claims to inspectable passages and explicitly acknowledging when sources cannot answer a query.**

**Q4.** **GraphRAG's indexing stages of entity and relationship extraction, hierarchical community detection, and community summary generation supply global context that nearest-passage RAG models miss.** **This global approach supports the archive-grounding aim in Document C—a course-summary substitute rather than the missing technical archive—by providing comprehensive corpus-wide synthesis and mitigating downstream risk.**

**Q5.** **Document C, explicitly identified as a course-summary substitute rather than the missing original technical archive, states that a dual-host Audio Overview evaluation is assigned as an educational activity to illustrate grounding principles.** **However, the sources explicitly confirm that audio debriefs do not provide empirical evidence of perfect citation accuracy, nor do they establish that every generated claim or citation is guaranteed correct.**

🔍 **Would you like me to draft a comprehensive report or structured comparison matrix synthesising the retrieval architectures, evaluation criteria, and grounding trade-offs across all three documents?**