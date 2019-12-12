# commen-sense-storytelling
Linking images by knowledge graph

```
To run the code, see run.sh
```

## Requirement
spacy with [neuralcoref](https://github.com/huggingface/neuralcoref)
gender-guesser

## Data Preprecossing
Prepossing source codes are in `data/preprocess_src`.
The pipline is:
1. coref_resolution.py
```
This code is to conduct correference resolution for stories
```
2. replace_coref.py
```
Then, replacing pronoun by its root noun
```
3. ner_gender.py
```
Finally, replacing NAMEs with [female, male]
```

This repository is mainly based on [jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
