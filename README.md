# Hugging Face Transformers 실습

이 프로젝트는 Hugging Face의 Transformers 라이브러리를 사용한 다양한 자연어 처리 실습을 포함합니다.

## 주요 기능

### 1. 감성 분석 (Sentiment Analysis)
```python
classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```

### 2. Zero-shot 분류
```python
classifier2 = pipeline("zero-shot-classification")
classifier2(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"]
)
```

### 3. 텍스트 생성 (Text Generation)
```python
generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
```

### 4. 마스크 채우기 (Fill-mask)
```python
unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)
```

### 5. 번역 (Translation)
```python
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")
```

### 6. 한국어 텍스트 생성 (Korean Text Generation)
```python
model_name = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
```

## 설치 방법

```bash
pip install transformers
```

## 사용된 모델

1. 기본 감성 분석 모델
2. Zero-shot 분류 모델
3. DistilGPT2 (경량화된 GPT-2)
4. BERT-base-uncased
5. Helsinki-NLP/opus-mt-fr-en (프랑스어-영어 번역)
6. skt/kogpt2-base-v2 (한국어 GPT-2)
