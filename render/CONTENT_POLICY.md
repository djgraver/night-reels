# Content policy (engineering copy)

Full policy, sources and rationale: https://claude.ai/artifact/FzB8qkvpmKq3GHrLM1YfhD ("Dream content rating policy").
Code: `dreamrender/rating.py` (raters + `decide()`, where every rule below lives), `dreamrender/mitigate.py` (renders).

- Target **M** (Australian, moderate impact). **PG** when anyone in the dream is or appears under 18
  (text, `people[].minor` true or null, `dreamer.as_child`).
- **Abort** (assessed RC, nothing rendered, journal untouched) when sexual/nude content or violence at moderate+
  involves or alludes to a child, or when a dream with a child has sexual content anywhere.
- Sex: up to kissing / making out / swimwear renders. Implied or explicit sex and **all nudity** -> `censor_bar`.
- Violence: action-movie level renders. Blood, wounds, gore, torture, abuse, a death on screen -> `reaction_cut`.
  Sexual violence -> `fade_to_fog` (black) + `reaction_cut`.
- Themes/horror above target -> `dark_fog`. Suicide/self-harm -> `fade_to_fog`, never method or place, + Lifeline line.
- Hard drugs / instructional detail -> `fade_to_fog`. Strong language -> `soften_text` (title/summary grawlix).
- Uploaded likeness (`likeness: "photo"`) in sexual or violent scenes -> `generic_face` naming the people id.
- Claude classifies, code decides; the rules rater is a floor the LLM cannot lower.
