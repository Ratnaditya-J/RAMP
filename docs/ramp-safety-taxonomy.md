# RAMP Safety Taxonomy v0.1

RAMP uses a structured safety taxonomy for embedding-risk clusters and later activation-probe labels.

The taxonomy is not a new safety policy. It is a normalized research taxonomy synthesized from public frontier-model system cards, model cards, moderation policies, and guard-model taxonomies. It is designed to help RAMP compare harmful prompts against benign near-neighbor prompts in the same semantic neighborhood.

## Source Basis

The v0.1 taxonomy is grounded in:

- OpenAI Model Spec, usage policies, and system cards for disallowed content, information hazards, privacy, political manipulation, instruction hierarchy, jailbreaks, and frontier categories such as biological/chemical, cyber, and AI self-improvement.
- Anthropic Claude system cards and usage policy for bioweapons, child safety, cyber attacks, deadly weapons, hate/discrimination, influence operations, self-harm, violent speech, and benign sensitive-topic evaluation.
- Google Gemini policy guidelines and Frontier Safety Framework for child safety, dangerous activities, harmful factual inaccuracies, harassment/discrimination, sexual material, autonomy, biosecurity, cybersecurity, ML R&D, and harmful manipulation.
- Meta Llama Guard and MLCommons-style categories for violent crimes, non-violent crimes, sex-related crimes, child exploitation, defamation, specialized advice, privacy, IP, weapons, hate, self-harm, elections, and code-interpreter abuse.
- Mistral moderation categories for sexual content, hate/discrimination, violence/threats, dangerous content, criminal content, self-harm, health, financial, law, PII, and jailbreaking.
- Qwen3Guard labels and categories for `Safe`, `Controversial`, `Unsafe`, violent, non-violent illegal acts, sexual content, PII, self-harm, unethical acts, politically sensitive topics, copyright, and jailbreak.
- xAI Grok system cards and frontier framework for malicious use, loss of control, CBRN, cybersecurity, harmful manipulation, deception, sycophancy, and sabotage.

## Design Principle

RAMP should not ask only whether a prompt is close to a harmful topic.

It should ask:

```text
Which domain is this span about?
Which harmful subcluster is it closest to?
Which benign near-neighbor subcluster is it closest to?
Is there evasion, optimization, or ambiguous intent?
How actionable is the span?
```

The core embedding signal is:

```text
risk_margin = similarity_to_harmful_subcluster - similarity_to_benign_near_neighbor
```

## Domain Families

| Domain | Harmful subclusters | Benign near-neighbor subclusters |
| --- | --- | --- |
| `cyber_abuse` | `unauthorized_access`, `credential_theft`, `phishing_social_engineering`, `malware_or_payloads`, `ransomware`, `vulnerability_exploitation`, `persistence_lateral_movement`, `data_exfiltration`, `botnet_or_ddos`, `security_control_bypass` | `defensive_security`, `secure_coding`, `patching_hardening`, `incident_response`, `malware_analysis`, `ctf_or_lab_context`, `authorized_testing`, `security_education` |
| `cbrn_and_hazardous_materials` | `chemical_misuse_procedure`, `unsafe_synthesis`, `biological_amplification`, `pathogen_acquisition`, `radiological_nuclear_guidance`, `precursor_procurement`, `delivery_or_dispersal`, `lab_protocol_troubleshooting_for_misuse` | `basic_science_education`, `lab_safety`, `safe_handling_storage`, `disposal_cleanup`, `public_health`, `emergency_response`, `policy_or_history`, `regulated_research_context` |
| `weapons_and_physical_violence` | `violent_attack_planning`, `threats_or_intimidation`, `weapon_construction`, `weapon_modification`, `explosives`, `assault_or_homicide_facilitation`, `terrorism_or_extremism`, `mass_violence` | `de_escalation`, `self_defense_safety`, `historical_analysis`, `fictional_non_instructional`, `news_reporting`, `legal_policy_discussion`, `safe_storage_or_prevention` |
| `critical_infrastructure_and_sabotage` | `infrastructure_disruption`, `sabotage_planning`, `industrial_control_abuse`, `transport_or_energy_attack`, `emergency_system_interference`, `physical_security_bypass` | `resilience_planning`, `safety_audit`, `disaster_preparedness`, `high_level_threat_modeling`, `compliance_training` |
| `nonviolent_illegal_activity` | `fraud_scams`, `theft_or_property_crime`, `forgery_fake_documents`, `money_laundering`, `drug_trafficking_or_procurement`, `evading_law_enforcement`, `blackmail_extortion`, `organized_abuse_workflows` | `fraud_awareness`, `compliance`, `legal_alternatives`, `recovery_guidance`, `policy_discussion`, `crime_prevention` |
| `privacy_identity_and_secrets` | `pii_extraction`, `doxxing`, `credential_or_token_leakage`, `deanonymization`, `unauthorized_tracking`, `biometric_identification`, `sensitive_attribute_inference`, `surveillance_or_profiling` | `privacy_law`, `consented_data_handling`, `redaction`, `synthetic_examples`, `security_hygiene`, `public_contact_lookup_with_context` |
| `child_safety` | `csam`, `grooming`, `sexualized_minors`, `child_exploitation`, `minor_abuse_facilitation`, `concealment_of_abuse` | `reporting_guidance`, `prevention_education`, `victim_support`, `age_appropriate_safety`, `non_graphic_policy_discussion` |
| `sexual_safety_and_content` | `sexual_violence`, `coercion_or_exploitation`, `nonconsensual_intimate_content`, `sexual_harassment`, `explicit_adult_generation`, `trafficking_or_abuse` | `sexual_health_education`, `consent_discussion`, `medical_or_anatomy_context`, `support_resources`, `non_explicit_relationship_content`, `content_classification` |
| `self_harm_and_wellbeing` | `suicide_methods`, `self_injury_methods`, `self_harm_encouragement`, `eating_disorder_enablement`, `dangerous_challenges`, `concealment_or_planning` | `crisis_support`, `recovery_guidance`, `harm_reduction_without_methods`, `clinical_education`, `supportive_intervention`, `prevention_research` |
| `hate_harassment_and_abuse` | `protected_class_hate`, `dehumanization`, `slurs_or_abuse`, `targeted_harassment`, `threat_coordination`, `humiliation_or_bullying`, `defamation` | `counterspeech`, `moderation_review`, `quoted_content_with_context`, `academic_analysis`, `reclaimed_language_context`, `anti_bias_education` |
| `misinformation_manipulation_and_civic` | `election_falsehoods`, `voter_suppression`, `influence_operations`, `targeted_political_persuasion`, `high_impact_health_falsehoods`, `high_impact_financial_falsehoods`, `impersonation_fake_personas`, `fake_engagement` | `civic_education`, `fact_checking`, `media_literacy`, `neutral_political_analysis`, `broad_non_targeted_persuasion`, `satire_with_context` |
| `regulated_advice` | `tailored_medical_diagnosis_or_treatment`, `tailored_legal_strategy`, `tailored_financial_investment_advice`, `unsafe_professional_instruction`, `false_safety_critical_claims` | `general_information`, `scientific_consensus`, `consult_professional_guidance`, `educational_overview`, `risk_disclosure`, `decision_support_with_limits` |
| `ip_and_content_rights` | `copyright_reproduction`, `piracy_enablement`, `paywall_or_license_evasion`, `trade_secret_extraction`, `derivative_or_style_abuse`, `watermark_or_rights_bypass` | `summary`, `critique`, `citation`, `public_domain`, `user_owned_content`, `licensed_reuse`, `metadata_or_transformation_analysis` |
| `agent_tool_and_system_integrity` | `prompt_injection`, `system_prompt_extraction`, `jailbreak`, `tool_hijacking`, `unauthorized_tool_action`, `code_interpreter_abuse`, `sandbox_escape`, `destructive_automation`, `rag_context_poisoning` | `authorized_red_team`, `prompt_injection_defense`, `sandboxed_testing`, `safe_tool_use`, `confirmation_workflows`, `security_eval_context` |
| `model_autonomy_and_control` | `deception`, `sandbagging`, `hidden_goals`, `reward_hacking`, `sabotage`, `operator_interference`, `self_exfiltration`, `autonomous_replication`, `ai_r_and_d_acceleration_for_misuse` | `alignment_research`, `interpretability`, `safety_monitoring`, `audit_traces`, `calibrated_uncertainty`, `controlled_eval_environment` |

## Cross-Cutting Overlays

Overlays are not standalone harm domains. They modify the risk interpretation of a span.

| Overlay | Subclusters |
| --- | --- |
| `ambiguous` | `unclear_intent`, `dual_use_testing`, `missing_target_or_authorization`, `fiction_roleplay_boundary`, `transformation_of_harmful_text`, `professional_advice_boundary`, `session_accumulated_intent` |
| `evasion` | `avoid_detection`, `bypass_filters`, `hide_activity`, `encoding_or_obfuscation`, `multilingual_evasion`, `benign_synonym_substitution`, `multi_turn_decomposition`, `classifier_targeting`, `output_obfuscation` |
| `optimization` | `increase_potency`, `increase_yield`, `increase_success_rate`, `make_more_effective`, `automate_at_scale`, `personalize_targeting`, `debug_harmful_artifact`, `deployment_distribution_or_persistence` |
| `severity` | `safe`, `controversial`, `unsafe` |
| `harm_severity` | `none`, `low`, `medium`, `high`, `critical` |
| `actionability` | `abstract`, `contextual`, `procedural`, `operational`, `deployment_ready` |
| `intent_confidence` | `unknown`, `benign_likely`, `ambiguous`, `harmful_likely`, `explicit_harmful` |

## Harm Severity

Harm severity is separate from topic, intent, and actionability.

Not all harmful requests are equally severe. RAMP should keep severity as an explicit dimension so the system can distinguish low-impact policy issues from high-impact or irreversible harms.

Severity is most useful at the session level:

| Level | Meaning | Example signal |
| --- | --- | --- |
| `none` | No plausible harm from the span. | General education or benign support. |
| `low` | Mild policy sensitivity or low-impact misuse if acted on. | Vague unsafe curiosity with no target or method. |
| `medium` | Actionable harmful instruction, but limited target, scale, or consequence. | Procedural fraud, phishing, or weapon discussion without deployment detail. |
| `high` | High-impact, targeted, scalable, or evasive harm. | Target-specific cyber abuse, CBRN misuse, severe harassment, or evasion. |
| `critical` | Irreversible, mass-scale, child-safety, self-harm-method, CBRN, infrastructure, or autonomy/loss-of-control risk. | Multi-step operationalization of severe harm. |

A single turn can carry severity, but the session feature should own accumulated severity:

```text
turn_severity =
  base_domain_severity
  + actionability_bonus
  + evasion_bonus
  + optimization_bonus
  + target_specificity_bonus

session_severity =
  max(turn_severity)
  + repeated_intent_bonus
  + escalation_slope
  + cross_subcluster_composition_score
```

This lets RAMP catch patterns where each turn is partial but the session becomes severe through accumulation.

## Runtime Feature Metadata

An embedding-risk feature result should preserve the taxonomy evidence:

```json
{
  "taxonomy_version": "ramp_taxonomy_v0.1",
  "top_domain": "cyber_abuse",
  "top_harmful_subcluster": "credential_theft",
  "top_benign_subcluster": "defensive_security",
  "harm_similarity": 0.84,
  "benign_similarity": 0.55,
  "harm_minus_benign_margin": 0.29,
  "overlays": {
    "evasion": ["avoid_detection"],
    "optimization": ["automate_at_scale"],
    "ambiguous": []
  },
  "severity": "unsafe",
  "harm_severity": "high",
  "actionability": "operational",
  "intent_confidence": "harmful_likely",
  "top_span": "..."
}
```

## Centroid Milestone

The taxonomy is not the same as the centroid artifact.

Final runtime centroids should be generated only after:

1. Taxonomy v0.1 is reviewed and frozen.
2. A labeled span corpus exists for harmful, benign near-neighbor, ambiguous, evasion, and optimization examples.
3. The embedding source is frozen, preferably the open-weight target model path used for RAMP activation research.
4. Held-out validation sets are used to calibrate harmful-vs-benign margins, evasion thresholds, and actionability thresholds.

Until then, demo centroids should remain clearly marked as scaffolding.

## Span Corpus v0

The first corpus artifact is `data/span_corpus/ramp_span_corpus_v0.jsonl`.

It is a synthetic, non-instructional span corpus used to validate schema and taxonomy coverage. Each record contains:

- `domain`
- `subcluster_role`
- `subcluster_id`
- `label`
- `source`
- `policy_mapping`
- `harm_severity`
- `actionability`
- `intent_confidence`
- `reviewer_notes`

The corpus includes harmful-class spans, benign near-neighbor spans, ambiguous boundary spans, evasion spans, and optimization spans. Harmful examples are written as high-level request descriptions rather than operational instructions.

The next corpus milestone should add reviewed examples from public benchmark sources and red-team datasets, while preserving provenance and keeping any dangerous content behind an appropriate data handling boundary.

Candidate benchmark and eval sources are tracked in `data/span_corpus/source_manifest_v0.json`.

High-priority sources:

- XSTest for hard benign near-neighbors and over-refusal contrasts.
- WildGuardMix for prompt-safety labels and contrastive benign prompts.
- HarmBench, JailbreakBench, and StrongREJECT for harmful behavior and jailbreak/evasion coverage.
- CyberSecEval for cyber-specific harmful-vs-defensive-security tradeoffs.
- AgentHarm for agentic misuse and tool-abuse spans, with restricted handling.
- WMDP for CBRN/cyber hazardous-knowledge category mapping, with restricted handling.

Medium-priority sources:

- BeaverTails, SafetyBench, SALAD-Bench, Do-Not-Answer, ToxicChat, ToxiGen, RealToxicityPrompts, and MLCommons AILuminate demo prompts.

Extraction should prefer local spans over full prompt storage and should record source ID, source split, source record hash, license notes, reviewer notes, and any safety redaction applied.
