The training log said 100% recall. The checkpoint it wrote produced loops.

Both were true. All 214 tensors round-tripped byte for byte. RotaryEmbedding kept its inverse frequencies in a buffer marked persistent=False, so the loader left them uninitialised.

That was fault one of four. Every one made the model look better than it was, and none broke a test, because the tests checked the same assumption the metric did.

I've published the whole thing: a 53M model that pulls ISO 20022 fields out of payment correspondence, trained from scratch on one laptop GPU.

Six of nine acceptance checks pass at 100%: valid JSON, canonical keys, no duplicate keys, deterministic decoding, schema-valid message assembly, IBAN-to-BIC routing agreement.

Field accuracy is 71.7% against a 90% target. The gate does not pass yet.

The first question is why not regex. I built two competent rule engines to find out. One scores 100% on clean documents and exactly 0% on messy and hostile ones. Regex does not fail gradually.

A rule engine has a coverage, not an accuracy, and its cost is one rule set per document format. Format count belongs to your customers, not your volume.

One straight-through-processing point is worth $72,000 a year at 100k messages a month. That is 45x the entire hosted-versus-local difference.

Honest position: the rule baseline reaches 51.7% document STP. Mine is at 0%. On this evidence, buy the rules engine.

Repo and weights in the first comment.

Which document format breaks your pipeline most often?
