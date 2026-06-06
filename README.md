# radar
RADAR (Robust Adversarial-Resistant Detection with Adaptive Reasoning) is a hybrid architecture for detecting AI-generated text, designed to be robust against adversarial obfuscations and style mimicry.

## Variants Submitted to TIRA

First ensure that the models are in the HF_HOME:

```
python3 -c 'from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained("yusr9/radar-encoder-freeze", trust_remote_code=True); AutoModel.from_pretrained("yusr9/radar-encoder-freeze", trust_remote_code=True);'
python3 -c 'from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained("yusr9/RADAR", trust_remote_code=True); AutoModel.from_pretrained("yusr9/RADAR", trust_remote_code=True);'
python3 -c 'from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained("yusr9/radar-encoder-freeze-pan26", trust_remote_code=True); AutoModel.from_pretrained("yusr9/radar-encoder-freeze-pan26", trust_remote_code=True);'
python3 -c 'from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained("yusr9/radar-encoder-freeze-raid", trust_remote_code=True); AutoModel.from_pretrained("yusr9/radar-encoder-freeze-raid", trust_remote_code=True);'
```

```
tira-cli code-submission \
    --dry-run \
    --path . \
    --task generative-ai-authorship-verification-panclef-2026 \
    --dataset pan26-generative-ai-detection-smoke-test-20260330-training \
    --mount-hf-model yusr9/radar-encoder-freeze answerdotai/ModernBERT-large \
    --command '/usr/local/bin/radar --model yusr9/radar-encoder-freeze'


tira-cli code-submission \
    --dry-run \
    --path . \
    --task generative-ai-authorship-verification-panclef-2026 \
    --dataset pan26-generative-ai-detection-smoke-test-20260330-training \
    --mount-hf-model yusr9/radar-encoder-freeze-pan26 answerdotai/ModernBERT-large \
    --command '/usr/local/bin/radar --model yusr9/radar-encoder-freeze-pan26'

tira-cli code-submission \
    --dry-run \
    --path . \
    --task generative-ai-authorship-verification-panclef-2026 \
    --dataset pan26-generative-ai-detection-smoke-test-20260330-training \
    --mount-hf-model yusr9/radar-encoder-freeze-raid answerdotai/ModernBERT-large \
    --command '/usr/local/bin/radar --model yusr9/radar-encoder-freeze-raid'


```
