RAG-calibration(1)
=============
RAG를 이용하여 Phi-3 모델의 accuracy를 높이는 실험 진행 

## Overview
1) 이전에 작성한 verbalized 코드를 이용하여 더 다양한 subject에서 모델의 accuracy와 ECE를 측정
+ result-regeneration 코드: [https://github.com/macroboomin/result-regeneration](https://github.com/macroboomin/result-regeneration)

2) [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) 데이터셋 이용
3) [Index file for wikipedia](https://myscale-datasets.s3.ap-southeast-1.amazonaws.com/RQA/IVFSQ_IP.index) 이용
- FAISS 벡터스토어를 사용하여 문서의 임베딩을 저장하여 Retreiver 객체 생성
- Langchain을 이용하여 검색 쿼리와 연관 높은 문서들을 검색
- prompt에 답변에 참고할 content로 첨부

## File Overview
### hotpot_qa
- `retrieval.py`로 HotpotQA 데이터셋을 가져와 FAISS 벡터스토어로 문서의 임베딩을 chunk로 저장
- `faiss`에 .faiss와 .pkl 형태로 Retriever 객체 저장

### wiki
- `retrieval.py`로 Wikipedia의 인덱스 파일들을 가져와 FAISS 벡터스토어로 문서의 임베딩을 chunk로 저장
- `faiss_wiki`에 .faiss와 .pkl 형태로 Retriever 객체 저장

### verbalized.py
- Phi-3 모델에게 단순히 질문하고 output에서 Answer와 Confidence 도출

### rag.py
- 저장된 retreiver 객체를 이용하여 langchain을 이용하여 연관이 높은 문서들을 검색하여 llm 모델이 답변에 있어 참고할 수 있도록 함

### results
- verbalized(`verbalized_results`), HotpotQA(`rag_results`), Wiki(`rag_wiki_results`)을 사용하였을 때의 각 subject의 결과들과 metrics들의 csv 파일


## Dataset
- [MMLU](https://huggingface.co/datasets/cais/mmlu) (Multiple Choice): College Mathematics,Business Ethics,Professional Law,Computer Security,Anatomy,Astronomy
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (Open number)
