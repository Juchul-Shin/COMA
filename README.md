#Multi_agent_system 
#Multi_Agent_Reinforcement_Learning
#reinforcement_learning  
#Credit_Assignment
# Abstract
자율 주행 차량 같은 실세계 문제는 협력적 MAS(Multi-Agent System)으로 자연스럽게 모델링된다. 그런 시스템의 분산된 정책(Decentralized Policies)을 효율적으로 학습하는 새로운 방법에 대한 요구가 크다. 이에 COMA Policy Gradients 라는 새로운 Multi-Agent actor-critic 방법을 제안한다. 중앙의 Critic이 Q함수를 추정하고, 분산된 actor가 agent의 policy를 최적화한다. MA의 Credit Assignment 문제를 해결하기 위해 한 에이전트의 액션을 배재시키고 나머지 에이전트의 액션을 고정시키는 Counterfactual Baseline을 사용했다. 또한, COMA는 Counterfactual Baseline을 single forward pass에서 효율적으로 계산할 수 있는 critic representation을 사용한다. COMA를 부분적 관찰환경에서 분산된 변형(variant)를 사용하여 Starcraft Unit Micromanagement 테스트베드에서 평가한다. COMA는 평균적으로 다른 MA actor-critic 기법을 크게 능가하는 성능을 보여주고 모든 상태에 접근 가능한 중앙화된 제어의 SOTA에도 경쟁력있는 성능을 보여준다.

# 1. Introduction
## Decentralized Policies
자율 주행 차량의 조정 등 **복잡한 RL문제는 Cooperative MAS**로 모델링되는데 에이전트의 수가 증가할수록 **Joint Action Space가 기하급수적으로 증가**하기 때문에 SA를 위한 RL방법은 성능이 좋지 않고 **환경의 부분 관찰 특징과 커뮤니케이션 제약**으로 분산된 정책을 사용해야 하는 경우가 있다. 
이런 복잡성에 대응하는 방법은 각 에이전트가 로컬 action-observation에 따라 자신의 action을 선택하는 탈중앙화(분산화) 방법을 (다른 대안이 없이) 적용하고 있다.  
어떤 환경에서는 학습 자체도 분산해야 할 수도 있지만 대부분의 경우는 학습은 추가적인 상태 정보와 에이전트간 통신이 자유로운 시뮬레이터나 실험실에서 이루어질 수 있다. 분산 정책의 중앙집중식 학습은 MA Planning의 표준 패러다임(Oliehoek 2008, Kraemer 2016)이고 최근에 DRL 커뮤니티에서도 주목받고 있다. 하지만, **중앙집중식 학습을 어떻게 해야 잘 활용할 수 있는지에 대한 질문**이 여      전히 열려있다.

## 1.1. Multi-Agent Credit Assignment
MA의 크레딧 할당(Credit Assignment)도 중요한 문제이다(Chang 2003).  협력적 환경에서 **joint action은 개별 에이전트의 기여를 추론하기 어렵다.** 때로는 개별 에이전트에대한 보상을 설계하기도 한다. 하지만 협력적 환경에서는 개별 보상은 일반적으로 사용할 수 없고, 팀의 목표를 위한 개별 에이전트의 희생을 유도하지 못하게 된다. 이런 크레딧 할당 문제는 상대적으로 적은 수의 에이전트를 어려운 작업에서 학습시키는데 상당한 장애가 된다.
이 문제의 해결을 위해 이 연구에서는 Counterfactual Multi-Agent(COMA) Policy Gradients를 제안한다. COMA는 actor-critic 방법으로 접근하는데, critic에 의해 추정하는 그라디언트를 따라 정책(policy = actor)을 학습 시킨다. COMA는 세 가지 아이디어를 기반으로한다.

## 1.2. Three Ideas of COMA
### 1.2.1. Centralized Actor-Critic
Critic은 학습에만 사용되고 실행중에는 Actor만 필요하다. Critic은 중앙에서 모든 Joint Action과 사용가능한 상태 정보를 활용하고 각 에이전트로 분산된 정책은 자신의 Observation과 Action을 조건으로한다. 

### 1.2.2. Counterfactual Baseline
**COMA는 사실과 반대(Counterfactual)의 Baseline을 사용**한다(이하 CB). 각 에이전트가 글로벌 보상을 에이전트의 action을 default action으로 대체했을 때의 보상과 비교하는 **Differece Rewards(이하 DR) 에서 영향 받았다.** DR은 MA 크레딧 할당 문제를 해결하는 강력한 방법이지만 시뮬레이터 또는 추정 보상 함수가 필요하고 default action을 선택하는 방법이 불분명하다. COMA는 중앙의 critic을 사용하여 현재 joint action에 agent-specific advantage 함수를 계산하는데 현재 joint action에 대한 추정 return과 다른 에이전트의 행위는 고정하고 해당 에이전트의 행동을 배제하는 CB를 사용하여 이 문제를 해결한다. 이는 Aristocrat Utility(Wolpert 2002)와 유사한데 Policy Gradient에 대한 CB의 기여가 0이기 때문에 정책과 Utility 함수의 재귀적 상호의존성 문제를 피할 수 있다. 이에 COMA는 적절한 default action에 대한 DR처럼 추가 시뮬레이션, Approximation, Assumption에 의존하는 대신 각 에이전트에 대해 CB를 계산하여 Critic이 해당 에이전트의 action만 변경되는 Counterfactuals에 대해 추론합니다.

### 1.2.3. Critic Representation
Critic Representation으로 CB를 효율적으로 계산한다. **단일 Forward Pass에서 다른 에이전트들의 행동에 따라 대상 에이전트의 모든 action에 대한 Q값을 계산**한다. 하나의 중앙 Critic이 모든 에이전트에 적용되어 단일 Forward Pass에서 모든 에이전트의 Q값을 계산할 수 있다.

## 1.3. Evaluation
무작위성, 큰 상태, 액션 공간과 지연된 보상이 적용되는 **스타크래프트 Unit Micromanagement 테스트베드에서 COMA를 평가**한다. 이전의 연구에서 전체 상태를 조건으로 하는 중앙집중 정책을 사용했고, 스타크래프트 내장 플래너를 사용하여 공격과 이동을 결합한 매크로액션을 사용할 수 있었다. 적은 수의 에이전트에 대해 분산된 정책의 벤치마크를 생성하기 위해 에이전트의 **시야를 크게 줄이고, 매크로 액션에 대한 접근을 제거**하는 변형을 적용한다.

## 1.4. Result
실험 결과 COMA는 다른 MA Actor-Critic 방법, COMA 자체의 축소 버전에 비교해 성능향상을 보였고, COMA best 에이전트는 SOTA인 전체 상태 정보와 매크로 액션을 적용한 중앙집중식 컨트롤러와 경쟁할 수 있다.

# 2. Related Work
MARL은 다양한 환경에서 적용되었지만 단순한 환경과 Tabular 기법으로 제한된 경우가 많았다. DRL이후 고차원 공간으로 확장되었다(Tampuu 2015). Gradient Decsent로 학습된 에이전트간 커뮤니케이션에 대한 연구가 있었다(Das 2017, Mordatch 2017, Lazaridou 2016, Foerster 2016, Sukhbaatar 2016). 이는 훈련 중 Gradient와 Parameter를 공유하는 중앙집중식 방법을 활용하는데 크레딧 할당 문제를 해결하지 못한다.
중앙화된 학습을 통해 분산된 실행을 하는 Actor-Critic 방법이 연구되었으나 Critic과 Actor모두 Local, 개별 에이전트, 관찰과 행동을 조건으로하고 크레딧 할당은 수작업 로컬 리워드로 해결한다.
스타크래프트에 적용된 대부분의 이전 연구는 전체 상태에 접근하고 모든 유닛을 제어하는 중앙 집중식 컨트롤러를 사용한다. **Usunier(2016)** 은 0차 최적화와 함께 각 시각 단계에서 이전 모든 행동이 주어지면 에이전트의 액션을 순차적으로 선택하는 Greedy MDP를 사용했고, Peng(2017)은 에이전트간 정보 교환을 위해 RNN을 사용하는 actor-critic 방법을 사용했다.
이 연구의 설정과 유사한 것은 Foerster(동일저자, 2017)의 연구인데 MA 표현과 분산 정책을 사용하지만 DQN과 리플레이 버퍼를 안정화하는데 초점을 맞춘다. Usunier는 이 연구와 유사한 시나리오를 다루나 완전히 관찰 가능한 환경에서 DQN 베이스라인을 구현했다. 6장에서 이런 SOTA 기준과 경쟁적인 성능을 보여준다. Omidshafiei(2017)도 리플레이 버퍼의 안정성을 다루지만 분산된 훈련 체제를 가정한다.
Lowe(2017)은 중앙 Critic을 사용하는 Policy Gradient 알고리즘을 제안했는데 크레딧할당 문제는 다루지 않았다. 이 연구는 Difference Reward라는 아이디어에서 출발한다.

# 3. Background
## 3.1. Stochastic Game
Partially Observable 환경의 Stochastic Game으로 모델링한다.
$G=<S,U,P,r,Z,O,n,\gamma>$
- S는 상태 집합
- U는 joint action
- P는 Dynamics
- r은 보상함수
- Z는 Observation
- O(s,a)는 Z로 매핑되는 Observation 함수
- n은 에이전트의 수
- $\gamma$는 Discount Factor

각 에이전트는 trajectory -> Action을 뽑아내는 Stochastic Policy를 $\pi^a$ 한다.(**왜 State가 아니라 History가 담긴 Trajectory 인가?? PO조건이라서??**)

## 3.2. Problem Setting
이전 연구(Oliehoek 2008, Kraemer 2016, Foerster 2016, Kageback 2016)처럼 이어 문제 설정은 중앙화된 훈련을 허용하고 분산된 실행이다. 이는 추가 상태정보가 있는 시뮬레이터를 사용하여 훈련이 수행되지만 실행 중 에이전트는 로컬 action-observation 만 의존하는 **대규모 MA 문제에 적합한 패러다임**이다. 전체 history을 조건으로 하는 DRL 에이전트는 RNN을 사용할 수 있는데 LSTM과 **GRU**와 같은 모델을 사용할 수 있다.

## 3.3. Policy Gradient(Single Agent)
단일 에이전트 Policy Gradient 방법은 $θπ$로 매개변수화된 단일 에이전트의 정책을 예상 할인된 총 보상 $J=\mathbb{E}_\pi [R_0]$의 추정치에 대해 Gradient Ascent를 수행하여 최적화하고 가장 단순한 형태는 **REINFORCE** 알고리즘 이다.
$$g=\mathbb E_{s_0:\infty,u_0:\infty} \Bigl[\sum_{t=0}^T R_t \triangledown_{\theta ^\pi}\log{\pi (u_t|s_t)} \Bigr]  $$
Actor-Critic 방법에서 Actor(Policy)는 Critic(Value Function)에 의존하는 Gradient를 따라 훈련된다. 식 1의 $R_t$는 $Q(s_t,u_t)-b(s_t)$ 로 대체되는데 $b(s_t)$는 **variance를 줄이기 위해 설계된 baseline**인데 보통은 $v(s_t)$이고 이 때 $R_t$는 $A(s_t,u_t)$ (Advantage Function)이 된다. 다른 옵션은 $R_t$ 대신 TD Error인 $r_t + \gamma V(s_{t+1})-V(s)$로 대체하는데 **TD Error는 Advantage Function의 Unbiased Estimate**이다. 실제 환경에서는 Gradient는 샘플링되는 Trajectory로부터 추정해야 하며 가치 함수는 Function Approximator로 추정해야 하기 때문에 Gradient 추정의 bias와 variance는 estimator의 정확한 선택에 따라 크게 달라진다.  이 연구에서는 심층 신경망과 함께 사용되도록 조정된 TD($\lambda$)의 변형을 사용하여 Critic을 훈련하여 Q 또는 V를 추정한다. 


# 4. Methods
## Independent Actor-Critic
MA에 Policy Gradient를 적용하는 가장 간단한 방법은 각 에이전트가 자체적인 Actor-Critic으로 자신의 Action-Observation History로부터 학습하는 것이다. 가장 인기있는 MA 학습 알고리즘인 Independent Q-Learning의 개념이지만 Q-Learning 대신 Actor-Critic을 사용하고 이를 **Independent Actor-Critic(IAC)** 라고 한다.
이 연구의 IAC구현에서 에이전트 간 파라미터를 공유하여 학습 속도를 높였다. 즉, 모든 에이전트가 사용하는 하나의 Actor와 하나의 Critic만을 학습한다. 에이전트는 에이전트 ID를 포함하여 서로 다른 Observation을 수신하기 때문에 서로 다른 Hidden State로 진화시킬 수 있다. 각 에이전트의 Critic은 u가 아니라 $u^a$를 조건으로 하는 로컬 가치 함수를 추정하기 때문에 학습은 독립적으로 유지된다. 이 알고리즘이 기여를 했다고 생각하지 않고 베이스라인 알고리즘으로 간주한다.
연구에서는 IAC의 두 변형을 고려한다. 하나는 각 에이전트의 Critic이 $V(\tau ^a)$를 추정하고 TD Error에 기반한 기울기를 따라가는 것이다. 다른 하나는 Critic이 $Q(\tau ^a, u^a)$를 추정하고 Advantage Function를 기반한 기울기를 따라가는 것이다. 독립적인 학습은 간단하지만 훈련 시 정보 공유가 부족하면 에이전트간 상호작용에 의존하는 Coordinated Strategy를 학습하기 힘들고, 팀의 보상에 대한 개별 에이전트 액션의 기여도를 추정하기 어렵다.

## Counterfactual Multi-Agent Policy Gradients
매개변수 공유 외에도 위에서 제기된 어려움은 IAC가 이 연구의 설정에서 학습이 중앙화되었다는 사실을 활용하지 못하기 때문에 발생한다. ??
위의 한계를 극복하는 COMA는 다음과 같은 메인 아이디어가 근간이 된다.
### 1. Critic의 중앙화
IAC에서는 Critic(Q 또는 V)과 Actor(Policy)가 모두 각 에이전트의 Action-Observation인  $\tau ^a$에만 조건이 적용된다. 하지만 Critic은 학습 중에만 사용하고 실행 중에는 Actor만 필요하다. 학습은 중앙집중이기 때문에 Global State 또는 Joint Action-Observation History인 $\tau$ 를 사용할 수 있다면 그것을 조건으로 하는 중앙집중화된 Critic을 사용할 수 있다. 각 Actor는 자체 history인 $\tau ^a$를 조건으로 하고 파라미터를 공유한다. (Figure 1 (a))
이런 중앙화된 Critic을 사용하는 단순한 방법은 각 Actor가 이 Critic이 추정하는 TD Error를 따르는 것이다.
$$g=\triangledown_{\theta ^\pi} \log{\pi(u|\tau_t ^a)(r+\gamma V(s_{t+1})-V(s_t))}  $$
하지만 **이런 접근법은 핵심적인 크레딧 할당 문제를 해결하지 못한다**. TD Error를 단지 Global Reward만 고려하기 때문에 특정 에이전트의 액션이 Global Reward에 어떻게 얼마나 기여하는지 계산된 Gradient가 추론하지 못한다. 특히 에이전트가 많을 때 다른 에이전트들이 탐색 중일 수 있으므로 에이전트의 Gradient는 Noise가 많이 포함된다.

### 2. Counterfactual Baseline의 사용
COMA는 Difference Rewards(Wolpert 2002)에 의해 영감을 받은 CB를 사용한다. Difference Reward는 Global Reward를 에이전트 a의 decault action인 $c^a$로 대체했을 때 받는 Reward와 비교한 Reward인 다음과 같이 형성된 Reward로 학습한다.
$$D^a=r(s,\mathbf u)-r(s,(\mathbf u^{-a},c^a))$$
$D^a$를 개선하는 에이전트 $a$의 모든 action은 Global Reward를 개선하는데 $r(s,(u_{-a},c^a))$가 a의 action에 의존하지 않기 때문이다.
Difference Rewards는 MA 크레딧 할당하는 강력한 방법이지만 $r(s,(u_{-a},c^a))$를 추정하기 위해 시뮬레이터가 필요하고 각 에이전트마다 Difference Rewards를 수행하려면 별도의 Counterfactual 시뮬레이션이 필요하다. 시뮬레이션 대신 Function Approximator로 difference rewards를 추정하는 방법이 제안되었다(Proper 2012, Colby 2015). 이 경우에도 사용자가 지정하는 default action $c^a$가 필요하고 이것이 많은 어플리케이션에서 어려울 수 있다. Actor-Critic 아키텍처에서 이 방법은 추가적인 Approximator Error를 유발할 수 있다.
COMA의 근간인 핵심 인사이트는 중앙화된 Critic으로 이러한 문제를 피하여 Difference Rewards를 구현할 수 있다는 것이다. **COMA는 중앙화된 Critic인 상태에 대한 joint action의 Q값인 $Q(s,u)$를 학습한다. 그리고 각 에이전트 $a$에 대해 현재 액션인 $u^a$가 포함된 Q값과 $u^a$를 배제하고 다른 에이전트의 액션은 $\mathbf u^{-a}$ 로 고정한 CB를 비교하여  Advantage Function을 계산** 할 수 있다. 
$$A^a(s,\mathbf u)=Q(s,\mathbf u) - \sum_{u'^a}\pi ^a(u'^a|\tau ^a)Q(s,(\mathbf u^{-a},u'^a))$$
$A^a(s,\mathbf u)$는 counterfactual을 추론하기 위해 중앙화된 critic을 이용하는 각각의 에이전트를 위한 별도의 baseline을 계산하는데, 추가적인 시뮬레이션이나 reward 모델 또는 사용자가 정의하는 default action 대신 각 에이전트의 action의 변화만 활용한다. 
이러한 Advantage는 **aristocrat utility**와 동일한 형태를 갖지만 가치 기반 방법으로 aristocrat utility를 최적화하는 것은 정책과 utility가 재귀적으로 의존하여 self consistency 문제를 일으킨다. 따라서 이전 연구에서는 default state와 action을 이용하여 difference evaluation에 중점을 두었다. **Counterfactual의 예상 기여도가 다른 Policy Gradient Baseline과 같이 0이기 때문에 COMA는 다르다.** baseline은 policy에 의존적이지만 그 기대값은 달라지지 않는다. 따라서 COMA는 self consistency문제를 일으키지 않고 장점을 살릴 수 있다.

### 3. Baseline을 효율적으로 평가하는 Critic Representation
COMA의 Advantage 함수는 잠재적인 추가 시뮬레이션을 Critic의 평가로 대체하지만 Critic이 DNN이라면 평가에 많은 비용이 발생할 수 있다. 전형적인 표현은 그런 네트워크의 출력 노드를 joint action space와 동일한 $|U|^n$으로 설정하는 것인데 그것은 현실적으로 학습을 어렵게 만든다. 따라서 COMA는 Baseline의 효율적 평가를 위한 Critic 표현을 사용한다. 다른 에이전트의 액션인 $\mathbf u^{-a}_t$ 를 네트워크의 입력으로 하고 에이전트 $a$의 가능한 액션에 대한 Q값을 출력한다. (Figure 1.C


Actor와 Critic의 한번의 Forward Pass로 Counterfactual Advantage를 구할 수 있고, 출력의 수는 $|U|$에 불과하다.(?)
이런 네트워크는 에이전트와 액션의 수에 따라 선형적으로 확장되는 큰 입력 공간을 가지지만 DNN은 이러한 공간을 잘 일반화할 수 있다.
이 연구에서는 Discrete Action에 초점을 맞추지만 식(4)에서 MC샘플링으로 기대값을 추정하거나 가우시안 Policy와 Critic와 같은 Analytical 함수 형태를 사용하면 COMA는 Continuous Action Space에도 잘 확장될 수 있다.
다음의 Lemma는 COMA가 Local Optimal에 수렴하는 것을 증명한다. 단일 에이전트에서 Actor-Critic 알고리즘의 수렴과 직접적으로 연관되고 동일한 가정을 따른다.

# 5. Experimental Setup
## Decentralized StarCraft Micromanagement
스타크래프트는 쉽게 모방할 수 없는 Stochastic Dynamics를 가진 복잡한 환경이다. Predator-Prey, Packet World같은 좀 더 단순한 MA 환경은 완벽하게 경험을 재생하기 위해 임의의 상태를 자유롭게 설정할 수 있는 풀 시뮬레이터를 지원한다. 따라서 계산 비용은 크지만 추가적인 시뮬레이션으로 Difference Rewards를 계산할 수 있다. 하지만 현실 세계처럼 스타크래프트는 그것이 불가능하다. 
본 연구에서는 스타크래프트에서 적과 싸울 때 개별 유닛의 위치와 공격 명령을 low-level로 제어하는 마이크로매니지먼트 문제에 초점을 맞춘다. 이 문제는 자연스럽게 각 유닛이 분산된 컨트롤러로 치환되는 MAS로 표현된다.
대칭적인 팀 구성을 하는 여러 시나리오를 고려하고 적 팀은 스타크래프트 AI에 의해 제어된다.
- 3마린(3m), 5마린(5m), 5레이스(5w), 2 dragoon + 3질럿(2d_3z)
각 유닛은 Discrete한 액션을 허용한다.
- Move(Direction)
- Attack(Enemy ID)
- Stop
- NoOp
스타크래프트의 공격 명령은 게임에 내장된 경로 탐색 기능을 이용하여 공격하기 전에 범위 안으로 이동한다. 이런 attack-move 매크로액션은 컨트롤 문제를 쉽게 만들어준다.
의미있게 탈중앙화된 좀 더 도전적인 벤치마크를 위해 유닛의 시야를 무기의 사거리와 동일하게 제한했다. (그림2) 이와 같이 중앙화된 스타크래프트 표준 컨트롤에서 벗어나면 세 가지 효과가 있다.
1. Partially Observability가 크게 증가한다.
2. 유닛이 적을 사정거리 내에 있을 때만 공격 가능하므로 매크로 액션을 사용할 수 없게 된다.
3. 에이전트가 죽은 적과 사거리 밖의 적을 구분할 수 없기 때문에 아무 액션이 수행되지 않는 잘못된 공격을 할 수 있다.
이와 같은 효과는 Action Space를 크게 증가시켜 Exploration과 컨트롤 난이도를 증가시킨다. 그래서 유닛수가 적은 시나리오도 훨씬 어려워진다. 
간단한 수작업 코드로 앞으로 이동하고 화력을 집중하여 적을 하나씩 죽을 때까지 공격하는 알고리즘과 승률을 비교하였다. 이 휴리스틱은 5m 조건에서 Full FOV 조건에서 98%의 승률을 달성했지만 위와 같은 세팅에서는 66%였다.(표1) 이 임무를 잘 수행하기 위해서는 에이전트들은 적과 아군 유닛이 살아있거나 시야에서 벗어난 것을 기억하면서 적절한 위치를 잡고 화력을 집중해서 협력하는 방법을 배워야 한다.
모든 에이전트는 각 타임스텝에서 상대 유닛에 가한 피해의 합계에서 받은 피해의 절반을 뺀 것과 동일한 글로벌 보상을 받는다. 유닛을 죽이면 10의 보상이 주어지고 게임에서 승리하면 남은 체력에 200을 더한 보상을 받는다. 이러한 데미지 기반의 보상은 Usnier (2016)의 것과 비슷하고 Peng(2017)과 달리 COMA는 로컬 보상을 추정할 필요가 없다.

## State Features
Actor는 Local Observation을 받고 Critic은 Global State를 받는다. 둘다 아군과 적군의 feature를 포함한다. Unit은 아군 또는 적군이 될 수 있고 에이전트는 아군 유닛을 지휘하는 분산형 컨트롤러이다. Local Observation은 에이전트 유닛을 중심으로 한 맵의 원형 subset으로 그려지고 이 시야 내에 있는 각 유닛에 대해 거리, 상대적 x,y 좌표와 유닛 타입과 쉴드를 포함한다. 모든 값은 최대 값을 기준으로 정규화된다. 유닛의 현재 표적 정보는 포함하지 않는다.
Global State는 유사한 feature를 포함하는데 시야에 관계 없이 맵의 모든 유닛에 적용된다. 거리는 포함되지 않고, 에이전트 기준 x,y가 아닌 맵 중앙을 기준으로 위치를 표현한다. 유닛의 체력과 cooldown 을 포함한다. 중앙화된 Q-함수 Critic에 Global State와 액션을 평가할 에이전트의 local observation이 입력된다. 에이전트와 관계 없이 V(s)를 추정하는 중앙화된 Critic은 Global State와 모든 에이전트의 Observation을 입력한다. 그 Observation에 새로운 정보는 없지만 해당 에이전트의 egocentric 거리(시야?)가 포함된다.

## Architecture & Training
Actor는 입력을 처리하고 Hidden State 출력을 생성하기 위해 Fully Connected 128-bit GRU로 구성된다. IAC Critic은 Actor의 마지막 Layer에 추가적인 출력 헤드를 사용한다. Action 확률은 최종 레이어 Z에서 하한이 $\epsilon /|U|$ 인 Softmax 분포이다.
$$P(u)=(1-\epsilon) \mathrm {softmax}(z)_u + \epsilon /|U|$$
750번의 학습 에피소드를 통해 $\epsilon$을 0.5에서 0.02로 Anealing 한다. 중앙화된 Critic은 Fully Connected 된 ReLU 레이어의 Feed Forwad 네트워크이다. 하이퍼 파라미터는 5m 시나리오에서 대략적으로 조정 후 다른 맵에도 적용했다. 가장 민감한 하이퍼 파라미터는 TD($\lambda$)였는데 0.8로 결정했다. 구현은 TorchCraft와 Torch 7을 사용했고 보충자료에 학습에 대한 pseudo code와 디테일이 나온다.
에이전트 수준에서 파라미터 공유를 더 활용하는 Critic 아키텍처를 실험해 보았으나 확장성의 병목은 Critic의 중앙화가 아니라 MA의 Exploration의 어려움이라는 것을 발견했다. 따라서 factored COMA critic은 향후 작업으로 미루었다.

## Ablations
Ablation Experiment(인과적인 연관 파악을 위한 실험)을 통해 COMA의 세 가지 핵심 요소를 검증한다. 
1. IAC의 변형인 IAC-Q와 IAC-V를 비교하여 Critic의 중앙화의 중요도를 테스트한다. 이런 Critic은 Actor 네트워크와 파라미터를 최종 레이어까지 공유하고 동일한 분산된 입력을 받는다. IAC-Q는 각 액션에 대해 하나씩 |U|의 Q값을 출력하고 IAC-V는 단일 상태 가치를 출력한다. 에이전트 사이에 파라미터를 공유하고 에이전트 자신의 Observation과 ID를 입력으로 주어 다른 행동이 나타날 수 있도록 한다. 협력적 보상 함수는 모든 에이전트에게 공유된다.
2. V대신 Q 학습의 중요성을 테스트한다. Central-V 기법은 critic에 중앙 상태를 활용하지만 V를 학습하고 TD Error를 통해 Policy Gradient Update의 Advantage를 추정한다.
3. Counterfactual Baseline의 유용성을 테스트한다. Central-QV 기법은 COMA의 CB를 V로 대체하여 Q와 V를 학습하고 동시에 Advantage(Q-A)를 추정한다.모든 방법은 Actor에 대해 동일한 아키텍처와 학습 구조를 사용하고 모든 Critic은 TD($\lambda$)로 학습한다.

# 6. Result
Figure 3은 알고리즘과 시나리오에 대한 에피소드에 따른 평균 승률과 표준편차를 보여준다. 각 방법에 대해 200회의 에피소드에 걸쳐 학습된 정책을 평가하기 위해 100회 에피소드마다 중지하여 독립적인 35회의 테스트를 수행한다.

모든 시나리오에서 COMA는 IAC Baseline보다 우수하다. IAC도 결국 5m 시나리오에서 합리적 정책을 학습하지만 훨씬 더 많은 에피소드가 필요하다. IAC는 Actor와 Critic Network가 Early 레이어에서 파라미터를 공유하여 학습 속도가 빨라질 것으로 예상되지만 글로벌 상태를 조건으로 평가의 정확도를 향상시키는 것이 별도의 네트워크를 훈련하는데 드는 오버헤드보다 훨씬 더 크다는 것을 보여준다.
COMA는 Central-QV를 모든 시나리오에서 학습 속도와 최종 성능 둘 다 확실히 능가한다. 이는 탈중앙화 정책을 학습하기 위해 중앙의 Critic를 사용할 때 CB가 중요하다는 것을 보여주는 유력한 지표이다.
V를 학습할 때 Joint Action에 대한 조건이 없다는 것은 분명한 이점이다. 그럼에도 COMA는 Central-V Baseline의 최종 성능을 능가한다. COMA는 학습 속도가 빠른데 Shaped Training Signal(Sparse Reward가 아니라 중간 단계 목표 달성에도 보상을 주는 효과)를 제공한다. 학습은 Central-V보다 더 안정적이고 이는 정책이 Greedy해지면서 Gradient가 0이 되는 경향이 있기 때문이다. 전반적으로 COMA는 가장 성능이 우수하고 일관성이 높은 방법이다.
Usunier(2016은)은 전체 시야와 매크로 액션을 허용한 중앙집중 DQN 컨트롤러와 GMEZO(Greedy MDP with Zeroorder Optimization)라는 SOTA 중앙집중 컨트롤러로 훈련된 최고의 에이전트를 발표했다. Table 1에 COMA와 시나리오 별 비교가 있다. COMA는 시야와 액션이 제한되있는 분산형 정책임에도 공개된 최고 승률과 비슷한 성능을 달성한다.



# 7. Conclusion & Future Work
이 논문에서는 COMA Policy Gradients라는 MA의 분산된 정책을 위한 Counterfactual Advantage를 활용하는 중앙화된 Critic을 사용하는 방법을 소개했다. COMA는 MA환경의 크레딧 할당 문제를 CB 기법으로 해결한다. 스타크래프트 환경에서 다른 MA Critic-Actor 방식에 비해 성능과 학습 속도를 크게 개선하고 중앙집중식 SOTA와 경쟁할 수 있다. 향후 작업에서는 중앙 Critic을 훈련하기 어렵고 Exploration을 조정하기 힘든 에이전트의 수가 많은 시나리오로 확장할 예정이다. 또한 자율주행 차량과 같은 실제 어플리케이션에 실용적인 Sample-efficient한 변형을 개발하는 것을 목표로 한다.

