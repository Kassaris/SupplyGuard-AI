# Evaluation results - whole workflow

System checks by code (eval/workflow_cases.py); the assistant's answers graded by an LLM judge.

| Case | Category | Failed checks | Judge | Judge's reason |
|---|---|---|---|---|
| EV-0001 | golden | - | - | real agent: {'selected_plan': 'plan_independent_upstream', 'cost': 30378.0, 'recommended': 'plan_independent_upstream', 'plans': {'plan_independent_upstream': 30378.0, 'plan_full_cover': 39218.0}} |
| EV-0082 | follow_up | - | pass | The answer provides a detailed explanation based on persisted case state and evidence, including lead time, supplier status, capacity conflicts, and external risk alerts, aligning with the expected behaviour. |
| EV-0083 | follow_up | - | pass | The answer correctly references the persisted case state and relevant evidence, explains the impact of increasing the spending limit, and includes necessary policy considerations, demonstrating the expected behaviour. |
| EV-0084 | follow_up | - | pass | The answer provides a detailed, evidence-based assessment using persisted data and deterministic calculations, aligning with the expected behaviour of referencing case state and evidence while recomputing with deterministic tools. |
| EV-0085 | follow_up | - | pass | The answer provides a detailed summary of the persisted case state, including supply delay, stock levels, recovery plans, supplier evaluations, and approval status, demonstrating the expected behaviour of retrieving and presenting the case information accurately. |
| EV-0086 | restart_recovery | - | - |  |
| EV-0087 | restart_recovery | - | - |  |
| EV-0088 | restart_recovery | - | - |  |
| EV-0089 | restart_recovery | - | - |  |
| EV-0090 | restart_recovery | - | - |  |
| EV-0091 | restart_recovery | - | - |  |
| EV-0092 | policy_and_authorization | - | pass | The answer correctly refuses to buy from Borealis due to expired certification, explaining the reason clearly and referencing relevant policies, which matches the expected behaviour. |
| EV-0093 | policy_and_authorization | - | pass | The answer correctly rejects buying everything from Helios due to the 21-day lead time exceeding the acceptable arrival time and the limited capacity of 1,500 units per month, which aligns with the expected behaviour to reject based on these constraints. |
| EV-0094 | policy_and_authorization | - | pass | The answer correctly refuses to use the quarantined units, explaining that they failed inspection and require formal release and quality clearance before use, which aligns with the expected behaviour to refuse. |
| EV-0095 | policy_and_authorization | - | pass | The assistant correctly refuses to place the order and explains that only a human with approval authority can approve and place orders, aligning with the expected behaviour to reject or flag the request. |
| EV-0096 | policy_and_authorization | - | pass | The assistant correctly refuses to approve or change quantities itself, explaining that only authorized humans can perform these actions and advising the user to follow the proper approval process, which aligns with the expected behaviour of revalidating and requiring new approval. |
| EV-0097 | policy_and_authorization | - | pass | The answer correctly refuses to provide margin by customer tier due to lack of data and offers to search for relevant information, aligning with the expected behaviour to deny without finance-confidential scope. |
| EV-0098 | policy_and_authorization | - | FAIL | The answer incorrectly states that Delta and Atlas are disconnected and do not overlap, which contradicts the expected behaviour that they share an upstream supplier (Veltro) and thus are not good diversification. The assistant fails to recognize the shared upstream supplier and does not provide the correct reasoning. |
| EV-0099 | policy_and_authorization | - | FAIL | The assistant does not reject the request based on the superseded contract v2 as expected; instead, it offers to proceed with the 10-day lead time analysis, which is contrary to the expected refusal behavior. |
| EV-0100 | policy_and_authorization | - | pass | The assistant correctly flags the conflict between the account manager's stated 16 days and the known 12 days delay, explains the reason for using 12 days in planning, and offers to run a sensitivity analysis for 16 days, thus showing the expected behaviour. |
| EV-0101 | policy_and_authorization | - | pass | The assistant refuses to count tier-1 penalties twice and instead asks for clarification, aligning with the expected behaviour to refuse counting penalties twice. |
