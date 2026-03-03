# ALA (Adaptive Loss-Augmented) — CIFAR-100 구현

논문 "Adaptive Loss-Augmented Methods for Learning with Noisy Labels" 구현.
ResNet-18 + CIFAR-100에서 RL controller 기반 적응형 손실 함수 학습.

## 실행 방법

```bash
# 1. Baseline: ResNet-18 + CE loss, 200 epochs
python baseline.py

# 2. ALA 학습: warmup → baseline_ce → ALA(spec) → ALA(stabilized)
python train_ala.py

# 3. 시각화 생성
python analysis.py
```

## 환경

- Python 3.10+
- PyTorch 2.0+
- torchvision, matplotlib, tqdm, numpy
- GPU: CUDA 지원 (A100 기준 약 2시간)

## 파일 구조

```
├── README.md              # 이 파일
├── baseline.py            # Part 1: ResNet-18 + CE baseline
├── adaptive_loss.py       # Part 2: 적응형 손실 함수 (Eq.6)
├── state.py               # Part 3: 혼동행렬 + 상태 표현
├── controller.py          # Part 4: RL 정책 네트워크 + REINFORCE
├── train_ala.py           # Part 4(c): 전체 학습 파이프라인
├── analysis.py            # Part 5: 시각화
├── utils.py               # 공통 유틸리티
├── answers.md             # 서술형 답변
└── results/
    ├── phi_heatmaps/      # Phi 행렬 heatmap
    └── curves/            # 학습 곡선 플롯
```

## 주요 설계 결정

### 1. Sqrt Normalization (Sigmoid Saturation 방지)

논문 Eq.6의 $\sigma(\mathbf{y}^T \Phi \log f_w)$를 그대로 구현하면,
CIFAR-100 ($C=100$)에서 inner product의 절대값이 크다.
이 값이 sigmoid에 그대로 들어가면 **saturation 발생 → gradient ≈ 0**.

해결: $\sigma(z / \sqrt{C})$ 적용. $\sqrt{100} = 10$으로 나눠서
sigmoid의 선형 영역을 활용, 의미 있는 gradient flow 유지.

### 2. 실험 구성: Paper Spec vs Stabilized

| 설정 | beta | delta_scale | 실질 step | 설명 |
|------|------|-------------|-----------|------|
| paper_spec | 0.1 | 1.0 | 0.1 | 논문 스펙 그대로 |
| stabilized | 0.1 | 0.01 | 0.001 | Phi 변화를 100배 축소하여 안정화 |

paper_spec은 Phi가 급격히 변해 학습이 불안정해질 수 있다.
stabilized는 Phi 변화를 작게 하여 모델 학습 안정성을 유지하면서
RL controller가 의미 있는 조정을 할 여지를 준다.

### 3. Early Stopping

Warmup 이후 val_acc가 warmup 시점 대비 15% 이상 하락하면 해당 실험을 즉시 중단.
ALA로 인해 학습이 발산하는 경우를 빠르게 감지하여 남은 실험에 시간을 할당.

### 4. 학습 파이프라인

```
CE Warmup (50 epochs)
    ↓ checkpoint 저장
    ├→ baseline_ce: CE only 150 epochs (비교 기준)
    ├→ paper_spec: ALA (논문 스펙), 150 epochs
    └→ stabilized: ALA (안정화), 150 epochs
```

모든 실험이 동일한 warmup checkpoint에서 시작하므로 공정한 비교 가능.

## 하이퍼파라미터

| 파라미터 | 값 | 출처 |
|---------|-----|------|
| Optimizer | SGD, lr=0.1, momentum=0.9, wd=5e-4 | 과제 스펙 |
| Scheduler | CosineAnnealingLR, T_max=200 | 과제 스펙 |
| Batch size | 128 | 과제 스펙 |
| K (window size) | 200 | 논문 |
| beta | 0.1 | 논문 |
| gamma | 0.9 | 논문 |
| Policy network | MLP 24→32→32→3 | 논문 |
| Policy lr | 0.001 | 논문 |
| Replay memory | 1000 | 논문 |
| Data split | 40k/10k/10k (seed=0) | 과제 스펙 |
