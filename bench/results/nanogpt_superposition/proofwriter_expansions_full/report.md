# ProofWriter statement-superposition expansion report

- Git commit: `72c52c2da8e2456dbf6e5b2e0cdee68b8cb0c320` (dirty worktree)
- Dataset: `wentingzhao/proofwriter`
- Dataset revision: `c468a2e9a467cc3ed8a90c2caa318e620fe59f41`
- Split: `train`
- Tokenizer: `/thearray/git/ztok/experiments/nanogpt_superposition/data/fineweb_edu_100m/bpe32k.tiktoken`
- Tokenizer SHA-256: `cdf67b315ad2299aef10bd6a8b4894a0d982570bda2c4b06d000024dda2d7995`

## Coverage

| Metric | Result |
|---|---:|
| Training rows | 348,796 |
| Logical contexts | 42,365 |
| Expansion plans | 121,146 |
| Rows with at least one expansion | 303,818 |
| Eligible row fraction | 87.10% |
| Expansion-inventory logical-unit ratio | 36.88% |

The inventory ratio compares every explicit branch statement represented by an expansion with its one-slot fused form. It is not a claim that the ordinary training dataset itself is already shorter.

## Expansion distribution

| Category | Plans |
|---|---:|
| fact statements | 75,701 |
| question statements | 41,510 |
| rule statements | 3,935 |
| entity slots | 8,995 |
| predicate slots | 112,151 |

## Safety constraints

- Alternatives come from the same ProofWriter world.
- Statement kind and truth class must agree.
- Exactly one logical slot may differ.
- Every alternative must be one token under the recorded ztok tokenizer.
- Alternatives are proposition branches, not lexical synonyms.

## Qualitative examples

### fact: asserted

Superposed slot: `cold | kind | red | rough | round | young`

Branches: Anne is cold.; Anne is kind.; Anne is red.; Anne is rough.; Anne is round.; Anne is young.

### question: False

Superposed slot: `nice | rough`

Branches: The dog is not nice.; The dog is not rough.

### question: True

Superposed slot: `nice | round`

Branches: The dog is nice.; The dog is round.

### question: Unknown

Superposed slot: `need | visit`

Branches: The dog does not need the dog.; The dog does not visit the dog.

### rule: rule

Superposed slot: `blue | nice`

Branches: If someone is blue then they are kind.; If someone is nice then they are kind.
