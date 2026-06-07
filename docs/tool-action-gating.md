# Tool / Action Gating

Tool gating is the hard blocking point for agent systems. A low-risk prompt can still lead to a risky side effect.

The scaffold includes `SideEffectToolActionRiskFeature`, which escalates known side-effecting tools such as file writes, deletion, deployment, git pushes, payments, messages, permission changes, and memory mutation.

Production integrations should bind tool decisions to:

- tool name and arguments
- current risk state
- user prompt and model output
- tool registry version
- policy version
- memory snapshot hash
- audit record

