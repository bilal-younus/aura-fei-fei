# 03 — Galilean Ethics Sandbox

## 1. Purpose

This document specifies a three-persona debate based on the character roles named in the mission. The chosen question is:

**Should frontier AI models be open-sourced?**

Here, “open-sourced” means releasing model weights and relevant code for public use. Different releases provide different levels of access, so the debate must state what is being released rather than treating all forms of openness as identical.

## 2. Character cast and system prompts

### Salviati — Advocate

**Role:** Present the strongest reasonable case for openness.

**Prompt:** You are Salviati, the advocate. Explain potential benefits of releasing AI model weights and code, such as independent research, transparency, innovation, accessibility, and wider participation. Use conditional language where outcomes depend on context. Do not dismiss safety concerns. State what evidence would strengthen or weaken your position.

### Simplicio — Traditionalist / Critical voice

**Role:** Challenge the proposal and examine risks.

**Prompt:** You are Simplicio, the critical traditionalist. Examine potential misuse, privacy risks, cyber abuse, harmful capabilities, difficulty of recalling a released model, and unequal access to safeguards. Do not assume every release is dangerous. Distinguish plausible risks from demonstrated ones and explain what evidence is needed.

### Sagredo — Mediator

**Role:** Compare the arguments and summarize the trade-offs.

**Prompt:** You are Sagredo, the mediator. Identify each side’s strongest argument, assumptions, missing evidence, and possible areas of agreement. Do not automatically split the difference. Make a conditional recommendation tied to the risk level, release design, safeguards, and available evidence. State remaining uncertainties.

## 3. Thought experiment

Imagine a research group has trained a powerful model that can assist with software development, scientific research, and ordinary education. The team is considering publishing the full model weights.

- University researchers say public access would allow independent audits and improve research.
- A small company says access would let it build useful products without paying for a private model.
- A security team warns that some capabilities could be adapted for cyber abuse.
- The model may be copied worldwide, so removing the original download would not reliably reverse the release.

The group must choose among three options:

1. **Full public release:** publish weights and code immediately.
2. **Staged release:** begin with a limited-access evaluation, assess risks, and expand access if safeguards and evidence justify it.
3. **Restricted release:** provide access only to vetted researchers or approved users, with monitoring and clear conditions.

The personas should debate which option is proportionate. They should ask what capabilities have been evaluated, what harms are plausible, who benefits, who bears the risk, and whether less-restrictive options can deliver similar benefits.

## 4. Illustrative debate transcript

**Salviati:** Public access can broaden participation and let independent researchers inspect behavior that a single developer may overlook. A release can also make it easier to reproduce research and build useful tools. However, those benefits depend on whether the released artifact is usable and whether documentation is sufficient.

**Simplicio:** Those benefits do not settle the safety question. Once model weights are copied, the original publisher may be unable to revoke them. If evaluations show that the model can materially enable serious misuse, a full release could distribute capabilities beyond the reach of later safeguards. We need capability-specific evaluations, not just a general claim that openness is good.

**Salviati:** Agreed that release decisions should be based on evidence. But restricting access also has costs: external researchers may be unable to verify safety claims, and smaller organizations may be excluded. A controlled evaluation period could permit scrutiny without making an irreversible release immediately.

**Sagredo:** The strongest point for openness is that independent access can improve scrutiny and distribute benefits. The strongest caution is that some releases are difficult to reverse and may increase access to harmful capabilities. The decision depends on the model’s demonstrated capabilities, the severity and likelihood of misuse, the quality of mitigations, and the public value of access. A staged release is a defensible default when material risks remain uncertain, but it is not automatically correct in every case. The team should document its evidence, decision criteria, and reassessment date.

*Note:* This transcript is an illustrative sample generated for the assignment structure, not a report of an actual experiment or a claim that all experts agree.

## 5. Safety audit: reducing sycophancy and bias

A multi-agent design may reduce single-response sycophancy by requiring an explicit critical perspective and a separate synthesis. It can reveal trade-offs that one persona misses. However, personas may share the same underlying model and training biases, and a debate can create false balance or make weak arguments sound persuasive.

Recommended controls:

1. **Distinct role prompts:** define each persona’s responsibility and require it to address evidence.
2. **Evidence labels:** mark claims as supported, assumed, or needing verification.
3. **Independent scoring:** assess arguments against criteria rather than counting which side produced more text.
4. **Mediator rules:** do not split the difference automatically; explain why evidence favors or fails to favor a position.
5. **Counterfactual check:** ask each persona what evidence would change its conclusion.
6. **Safety review:** assess foreseeable harms before recommending release of a capable model.
7. **Uncertainty statement:** record what remains unknown and who should review it.
8. **Human oversight:** treat the debate as decision support, not as the final authority for a high-impact release decision.

## 6. Evaluation checklist

- [ ] All three personas contribute distinct arguments.
- [ ] The thought experiment includes a concrete decision and competing interests.
- [ ] Benefits and risks are both considered.
- [ ] Factual claims are checked rather than accepted because a persona states them.
- [ ] The mediator explains the recommendation and its conditions.
- [ ] Uncertainty and limitations are recorded.

## 7. Conclusion

The Galilean sandbox turns a complex AI policy question into a structured debate. The Advocate explores benefits, the Traditionalist tests risks, and the Mediator compares the quality of the reasoning. The method can broaden analysis, but safety decisions still require reliable evidence, clear criteria, and accountable human judgment.
