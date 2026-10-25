# Course website FST test

Date: 25 October 2026. Site: https://dlearn-cai.web.app/ -> Week 2 -> 3. Simulation.

The website is a separate reference implementation, not a runner for uploaded Python code. Its slot names sector, warpFactor and clearanceCode correspond to our destination, speed and clearance_code.

| Input | Observed state | Memory |
|---|---|---|
| Hello AURA online | GREETING | empty |
| Orion Arm | COLLECT_SLOTS | sector=Orion Arm |
| Warp 7 | CONFIRMATION | warpFactor=Warp 7 |
| ALPHA-7 | TERMINAL_SUCCESS | clearanceCode=ALPHA-7 |

After Reset Transducer, the built-in failure example sent `Can you tell me a joke about space?` from INIT. The visible result was FAILURE_STATE with an unrecognized greeting error. Our implementation instead allows bounded recovery before aborting. The website requests the code in CONFIRMATION; our implementation collects the code before a separate confirmation.

Evidence: `evidence/website_success.txt`, `website_failure.txt` and corresponding PNG screenshots. These tests did not upload a solution, sign off roles, change squadron membership or submit the assignment. The visible terminal dropdown remained Squadron Lovelace; these are generic workbench tests, not a Fei-Fei submission record.
