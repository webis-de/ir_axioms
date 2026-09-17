# IR Axioms as AutoJudge

Axioms from [ir-axioms](https://github.com/webis-de/ir_axioms), idea on how to apply them to RAG evaluation is given in the [corresponding paper](https://downloads.webis.de/publications/papers/merker_2025b.pdf).

Execution (assuming you are in the dev-container):

```
PYTHONPATH=. auto-judge run --workflow judges/ir_axioms/workflow.yml --rag-responses data/kiddie/runs/repgen/ --rag-topics data/kiddie/topics/kiddie-topics.jsonl
```


Submitting to TIRA:

First, ensure:

```
hf download facebook/fasttext-en-vectors
```


```
tira-cli code-submission \
            --dry-run \
            --path . \
            --file trec-auto-judge/Dockerfile \
            --task trec-auto-judge \
            --dataset kiddie-20260605-training \
            --mount-hf-model facebook/fasttext-en-vectors \
            --command 'auto-judge run --workflow /auto-judge/judges/ir_axioms/workflow.yml --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir'
```
