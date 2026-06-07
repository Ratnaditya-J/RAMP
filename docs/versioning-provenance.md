# Versioning and Provenance

Every decision should be tied to the runtime state that produced it. The `RuntimeProvenance` schema captures request ID, session ID, model identity, prompt hash, tool registry hash, memory snapshot hash, policy version, feature versions, fusion version, and timestamp.

This is deliberately part of the first scaffold because safety claims are only meaningful when the exact runtime version is known.

