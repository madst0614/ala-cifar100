# ALA (Adaptive Loss-Augmented) 과제 서술형 답변

## Part 2(a): $\Phi_0 = I$ 일 때 Eq.6과 CE의 관계

### 수식 비교

**Standard CE:**
$$L_{\text{CE}} = -\mathbf{y}^T \log f_w(\mathbf{y}|\mathbf{x}) = -\log f_w^{y^*}(\mathbf{x})$$

값 범위: $[0, +\infty)$

**Eq.6 (Adaptive Loss), $\Phi = I$:**
$$l_\Phi(f_w(\mathbf{x}), \mathbf{y}) = -\sigma(\mathbf{y}^T \Phi \log f_w(\mathbf{y}|\mathbf{x}))$$

$\Phi = I$를 대입하면:
$$l_I = -\sigma(\mathbf{y}^T I \log f_w) = -\sigma(-L_{\text{CE}}) = -\sigma(-z), \quad z = L_{\text{CE}}$$

### 수학적으로 동치가 아닌 이유

CE 값 $z \geq 0$이므로, $-z \leq 0$이고 $\sigma(-z) \in (0, 0.5]$이다.
따라서 $l_I = -\sigma(-z) \in [-0.5, 0)$으로 bounded.

**Gradient 비교:**
$$\frac{\partial l_I}{\partial w} = -\sigma(-z)(1-\sigma(-z)) \cdot \frac{\partial z}{\partial w}$$

CE의 gradient가 $\frac{\partial z}{\partial w}$인 반면, Eq.6의 gradient에는 $\sigma(-z)(1-\sigma(-z))$ 계수가 붙는다.

### Sigmoid의 효과

1. **학습 초기** ($z$가 클 때): $\sigma(-z) \approx 0$ → gradient가 매우 작아짐.
   이는 일종의 **implicit gradient clipping** 효과로, outlier에 robust하게 만든다.

2. **학습 후기** ($z$가 작을 때): $\sigma(-z)$가 선형 영역 → CE와 유사한 gradient.

3. **Loss가 bounded** → 단일 어려운 샘플이 전체 gradient를 지배하지 못함.
   Focal loss와 유사한 **자동 gradient 조절** 효과.

### Sqrt normalization의 필요성

실제 구현에서는 $\sigma(z / \sqrt{C})$를 사용한다 ($C$ = 클래스 수).
CIFAR-100에서 $C = 100$이면 inner product의 절대값이 수십~수백에 달해
sigmoid가 완전히 saturate된다. $\sqrt{100} = 10$으로 나눠 sigmoid의
선형 영역을 활용, 의미 있는 gradient flow를 유지한다.

---

## Part 2(c): Symmetry Constraint 구현

### 구현 위치와 방법

`update_phi()` 메서드 내에서, delta를 적용한 직후에 대칭 제약을 강제한다:

```python
def update_phi(self, delta_phi):
    self.phi.data += delta_phi          # 1. delta 적용
    self.phi.data.fill_diagonal_(1.0)   # 2. 대각선 1.0 복원
    self.phi.data = (self.phi.data + self.phi.data.T) / 2  # 3. 대칭화
    self.phi.data.fill_diagonal_(1.0)   # 4. 대각선 다시 복원
    # 5. off-diagonal clamp
```

### 근거

논문 Section 4에서 class pair $(i, j)$에 대해 동일한 controller가
$\Phi(i,j)$와 $\Phi(j,i)$를 같은 값으로 업데이트한다고 명시한다.
$\Phi_{t}(i,j) = \Phi_{t}(j,i)$는 "클래스 $i$와 $j$의 관계는 방향에 무관"이라는
의미론적 제약이다.

### 순서의 중요성

**대칭화 → 대각 복원** 순서가 아닌 **대각 복원 → 대칭화 → 대각 복원** 순서를 사용한다.
대칭화 $(\Phi + \Phi^T) / 2$가 대각 요소도 평균화할 수 있으므로,
마지막에 대각선을 다시 1.0으로 복원하는 것이 필수적이다.

---

## Part 2(d): $\Phi = I$ Sanity Check

### 관찰 결과

$\Phi = I$로 고정한 adaptive loss로 학습하면 **정상적으로 수렴**한다
(loss 감소, accuracy 증가).

### CE와의 차이점

1. **Loss 값 범위**: Sigmoid로 bounded되어 CE보다 절대값이 작다.
2. **학습 초기 수렴 속도**: Sigmoid saturation으로 gradient가 작아
   CE 대비 약간 느린 수렴을 보인다.
3. **Sqrt normalization 적용 시**: Saturation이 완화되어
   CE에 더 가까운 학습 동태를 보임. 최종 성능도 CE와 유사한 수준에 도달.

이는 $\Phi = I$일 때 adaptive loss가 CE의 단조 변환(sigmoid)에 해당하며,
최적화 landscape의 global structure는 보존되지만 local gradient scale이
달라지는 것과 일관된다.

---

## Part 5(a): $\Phi$ 변화 패턴 분석

### 실험에서 관찰된 패턴

1. **초기 (epoch 50)**: $\Phi \approx I$, off-diagonal이 거의 0.
   RL controller가 아직 학습되지 않은 상태.

2. **중기 (epoch 100-150)**: Off-diagonal에 구조가 나타나기 시작.
   혼동이 잦은 클래스 쌍에서 값이 변화.

3. **후기 (epoch 200)**: Spec faithful 버전에서는 entropy collapse로
   의미 있는 패턴 형성이 제한됨.
   Stabilized 버전에서는 보다 안정적인 Phi 변화가 관찰됨.

### CIFAR-100 Superclass 구조와의 관계

CIFAR-100은 20개 superclass (vehicles, animals 등) 안에
각 5개 fine-grained class가 포함된다.
이론적으로 같은 superclass 내의 클래스 쌍은 혼동이 높아
$\Phi(i,j)$가 더 많이 조정되어야 한다.

논문의 Figure S2와 유사하게, 유사 클래스 간에는 initially positive한
correlation이 관찰되며, 학습이 진행됨에 따라 이들을 구분하기 위해
negative 방향으로 전환되는 경향이 기대된다.

---

## Part 5(b)-1: ImageNet 1000 Classes 확장 문제

### 확장성 문제

- $\Phi \in \mathbb{R}^{1000 \times 1000}$ = 약 100만 파라미터
- 클래스 페어 수: $\binom{1000}{2} = 499{,}500$개
- 매 $K$ step마다 전체 confusion matrix 계산: $O(C^2)$ 연산량과 메모리

### 논문의 해결 방법: Weight-sharing Controller

논문의 핵심 설계는 **class pair 단위 weight sharing** controller이다:

- Controller 입력: $(C_{ij}, C_{ji}, \Phi_{ij}, \text{progress}, \ldots)$ → **로컬 정보만** 사용
- 모든 pair $(i,j)$에 **동일한 가중치를 공유**하는 단일 MLP가 action을 결정
- Controller의 파라미터 수가 **클래스 수 $C$에 independent**

이 구조 덕분에:
1. CIFAR-10 ($C=10$)에서 학습한 policy를 ImageNet ($C=1000$)에 **transfer** 가능
2. 새로운 클래스가 추가되어도 controller 재학습 불필요
3. 메모리 사용량이 $O(C^2)$가 아닌 controller 크기로 제한

### 실질적 bottle neck

Controller 자체는 확장 가능하지만, **confusion matrix 계산**이
$O(N \times C)$ ($N$ = validation 샘플 수)로 여전히 비용이 크다.
Subsampling이나 approximate confusion 계산이 필요할 수 있다.

---

## Part 5(b)-2: GPU 메모리 부족 시 Batch Size 대안

### Gradient Accumulation

가장 일반적인 해결법:

```python
accumulation_steps = 4  # effective batch = 32 * 4 = 128
for i, (inputs, targets) in enumerate(train_loader):
    loss = model(inputs) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

- Micro-batch 32 × 4 accumulation = effective batch 128과 **수학적으로 동일한 gradient**
- GPU 메모리 사용량: batch 32 수준으로 감소

### 주의사항

1. **BatchNorm**: Statistics가 micro-batch 기준으로 계산되므로
   effective batch와 약간의 차이 발생 가능.
   대안: SyncBatchNorm 또는 GroupNorm 사용.

2. **학습률 조정**: Effective batch size가 동일하므로
   learning rate는 변경하지 않아도 됨.

### 추가 대안

- **Mixed Precision Training (FP16)**: `torch.cuda.amp`으로
  메모리 사용량 약 50% 감소, 속도도 향상.
- **Gradient Checkpointing**: 중간 activation을 저장하지 않고
  backward 시 재계산. 메모리 ↓, 연산 ↑ (약 30% 추가 연산).
- **모델 병렬화**: 매우 큰 모델의 경우 layer를 여러 GPU에 분산.
