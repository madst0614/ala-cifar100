# ALA 과제 서술형 답변

## Part 2(a): Φ₀ = I일 때 CE와의 차이

### Cross-Entropy Loss

표준 cross-entropy loss는 다음과 같다:

$$l_{CE} = -y^T \log f_w(x) = -\log p(\text{정답 클래스})$$

정답 클래스의 예측 확률 $p$가 1에 가까워지면 loss는 0에 수렴하고, $p$가 0에 가까워지면 loss는 $+\infty$로 발산한다. 따라서 CE의 값 범위는 $[0, \infty)$이다.

### ALA Loss (Φ = I)

Φ를 단위행렬 $I$로 설정하면:

$$l_{ALA} = -\sigma(y^T I \log f_w(x)) = -\sigma(\log p(\text{정답 클래스}))$$

$\log p$의 범위는 $(-\infty, 0]$이므로, $\sigma(\log p)$의 범위는 $(0, 0.5]$이다. 따라서 $l_{ALA}$의 범위는 $[-0.5, 0)$이다.

### 핵심 차이: sigmoid 함수

두 loss의 차이를 유발하는 핵심 요소는 **sigmoid 함수** $\sigma(\cdot)$이다.

1. **Loss 값의 bounded 특성**: CE는 잘못된 예측에서 loss가 무한대로 발산할 수 있지만, ALA는 sigmoid에 의해 항상 $(-1, 0)$ 범위에 묶인다. 이는 RL agent가 Φ를 조절할 때 loss 값이 폭발하지 않도록 하는 안전장치 역할을 한다.

2. **Gradient 특성의 차이**: CE의 softmax output에 대한 gradient는 $p - y$로 깔끔한 형태를 갖는다. 반면 ALA는 chain rule에 의해 sigmoid의 derivative $\sigma'(\cdot) = \sigma(\cdot)(1 - \sigma(\cdot))$가 추가로 곱해진다. 이 때문에 gradient가 attenuate되어, 특히 학습 초반에 CE보다 수렴 속도가 느릴 수 있다.

3. **RL 학습 안정성**: Φ가 변할 때 loss landscape가 급변하면 RL agent의 reward signal이 불안정해진다. Sigmoid가 loss를 bounded하게 만들어 Φ 변화에 따른 loss 변동 폭을 제한함으로써, RL controller가 안정적으로 학습할 수 있는 환경을 제공한다.

---

## Part 2(c): Symmetry Constraint

### 왜 Φ(i,j) = Φ(j,i)를 강제하는가

Φ 행렬에서 $\Phi(i,j)$는 클래스 $i$를 예측할 때 클래스 $j$의 log-probability에 부여하는 가중치이다. Symmetry constraint는 클래스 $i \to j$ 관계와 $j \to i$ 관계가 동일해야 한다는 것을 의미한다.

예를 들어, "고양이"와 "호랑이"가 유사한 클래스라면, 고양이를 학습할 때 호랑이 정보를 활용하는 정도와 호랑이를 학습할 때 고양이 정보를 활용하는 정도가 같아야 논리적으로 일관된다.

### 적용 시점

Symmetry는 `update_phi()`에서 `delta_phi`를 적용한 **직후**에 강제한다:

```python
self.phi.data += delta_phi
self.phi.data = (self.phi.data + self.phi.data.T) / 2  # symmetry
self.phi.data.clamp_(-1, 1)                             # clamp
```

Action 적용 **전**에 symmetry를 강제하면, 이후의 action이 symmetry를 깨뜨릴 수 있다. Action 적용 **후**에 강제하면 Φ가 항상 symmetric 상태를 유지하게 되므로, 적용 후가 올바른 시점이다.

### 구현

대칭화는 단순히 전치 행렬과의 평균으로 구현한다:

$$\Phi \leftarrow \frac{\Phi + \Phi^T}{2}$$

이 연산은 임의의 정사각 행렬을 대칭 행렬로 변환하며, 원래 대칭이었던 경우에는 값을 변경하지 않는다.

---

## Part 2(d): Sanity Check 관찰

### 실험 결과

Φ = I로 고정한 AdaptiveLoss를 사용하여 ResNet-18을 200 epoch 동안 학습한 결과:

- Loss가 epoch이 진행됨에 따라 꾸준히 감소하여, adaptive loss가 정상적으로 동작함을 확인하였다.
- Test accuracy도 epoch이 진행됨에 따라 증가하여, 모델이 유의미하게 학습되고 있음을 확인하였다.

### CE 대비 차이점

- **Loss 값 범위**: CE loss는 양수 범위 $[0, \infty)$에서 시작하여 0에 수렴하지만, adaptive loss는 음수 범위 $[-0.5, 0)$에서 움직인다. 따라서 loss curve의 절대값 자체는 직접 비교할 수 없다.
- **수렴 속도**: Sigmoid의 gradient attenuation 효과로 인해 학습 초반 수렴이 CE보다 느릴 수 있다. Sigmoid derivative가 추가로 곱해지면서 effective learning rate가 줄어드는 효과가 있기 때문이다.
- **최종 성능**: Φ = I인 경우 CE와 유사한 수준으로 수렴하지만, gradient attenuation으로 인해 약간의 성능 차이가 발생할 수 있다.
- **핵심 확인 사항**: Adaptive loss가 정상적으로 학습을 수행할 수 있으며, Φ를 변화시키면 loss landscape가 달라질 수 있다는 것이 sanity check의 핵심이다.
