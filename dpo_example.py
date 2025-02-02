# DPO 훈련 과정을 단계별로 설명해드리겠습니다:

# 모델 준비:


# 기본 모델 (SFT로 학습된 모델)을 로드합니다
# 참조(reference) 모델을 생성합니다 (SFT 모델의 복사본)
# 토크나이저를 설정합니다


# 데이터셋 형식:

# pythonCopy{
#     'prompt': ["질문1", "질문2", ...],
#     'chosen': ["선호하는 답변1", "선호하는 답변2", ...],
#     'rejected': ["비선호 답변1", "비선호 답변2", ...]
# }

# 주요 학습 설정:


# learning_rate: 일반적으로 5e-5 정도의 낮은 값 사용
# batch_size: 메모리에 맞게 조절 (예: 4 또는 8)
# beta: DPO 손실 함수의 강도를 조절하는 파라미터 (보통 0.1)
# max_length: 시퀀스 최대 길이


# 학습 과정:


# DPOTrainer가 선호/비선호 응답 쌍을 사용해 학습
# 참조 모델과 비교하며 선호도 차이를 최적화
# 정기적으로 체크포인트 저장


# 주의사항:


# 메모리 관리를 위해 gradient_accumulation 사용
# 학습 안정성을 위해 혼합 정밀도 훈련(mixed precision) 사용
# 체크포인트 저장 주기 설정 필요

# 이 방식으로 모델은 인간의 선호도에 더 잘 맞는 방향으로 최적화됩니다.
# 실제 구현 코드를 보고 싶으시다면 말씀해 주세요.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments
from trl import DPOTrainer

def train_dpo(
    base_model_name: str,
    dataset,  # 데이터셋 (prompt, chosen, rejected 컬럼 포함)
    output_dir: str = "dpo_model",
    num_train_epochs: int = 1,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 1024,
    beta: float = 0.1,  # DPO 손실 함수의 온도 파라미터
):
    # 모델과 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Reference 모델 로드 (SFT 모델의 복사본)
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        bf16=True,
        evaluation_strategy="steps",
        eval_steps=100,
    )
    
    # DPO Trainer 초기화
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        max_prompt_length=max_length // 2,
    )
    
    # 학습 시작
    dpo_trainer.train()
    
    # 모델 저장
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

# 사용 예시
if __name__ == "__main__":
    # 데이터셋이 다음과 같은 형식이라고 가정:
    # {
    #     'prompt': [...],
    #     'chosen': [...],  # 선호되는 응답
    #     'rejected': [...] # 선호되지 않는 응답
    # }
    
    base_model_name = "your_sft_model_name"
    trained_model, tokenizer = train_dpo(
        base_model_name=base_model_name,
        dataset=your_dataset,  # 여기에 실제 데이터셋을 넣으세요
        num_train_epochs=1,
        batch_size=4
    )