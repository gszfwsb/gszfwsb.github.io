# 第二章 Markov Decision Process (马尔可夫决策过程)

强化学习中，智能体与环境就是这样进行交互的，这个交互过程可以通过马尔可夫决策过程来表示，所以马尔可夫决策过程是强化学习的基本框架。

在介绍马尔可夫决策过程之前，我们先介绍它的简化版本：马尔可夫过程（Markov process，MP）以及马尔可夫奖励过程（Markov reward process，MRP）。通过与这两种过程的比较，我们可以更容易理解马尔可夫决策过程。其次，我们会介绍马尔可夫决策过程中的**策略评估（policy evaluation）**，就是当给定决策后，我们怎么去计算它的价值函数。最后，我们会介绍马尔可夫决策过程的控制，具体有**策略迭代（policy iteration）** 和**价值迭代（value iteration）**两种算法。在马尔可夫决策过程中，它的环境是全部可观测的。但是很多时候环境里面有些量是不可观测的，但是这个部分观测的问题也可以转换成马尔可夫决策过程的问题。

# 一、马尔可夫过程

MP定义 

- $S$ 有限状态集
- $P^a$ 动作转移模型，$P\left(s_{t+1}=s^{\prime} \mid s_t=s\right)$
- 折扣银子 $\gamma \in[0,1]$
- MDP 是一个元组: $(S, P, \gamma)$。

## 1.1 马尔可夫性质

在随机过程中，马尔可夫性质 (Markov property) 是指一个随机过程在给定现在状态及所有过去状态情况下，其末来状态的条件概率分布仅依赖于**当前状态**。以离散随机过程为例，假设随机变量 $X_0, X_1, \cdots, X_T$ 构成一个随机过程。这些随机变量的所有可能取值的集合被称为状态空间 (state space) 。如果 $X_{t+1}$ 对于过去状态的条件概率分布仅是 $X_t$ 的一个函数，则

$$
p\left(X_{t+1}=x_{t+1} \mid X_{0: t}=x_{0: t}\right)=p\left(X_{t+1}=x_{t+1} \mid X_t=x_t\right)
$$

其中， $X_{0: t}$ 表示变量集合 $X_0, X_1, \cdots, X_t ， x_{0: t}$ 为在状态空间中的状态序列 $x_0, x_1, \cdots, x_t$ 。马尔可夫性质也可以描述为给定当前状态时，将来的状态与过去状态是**条件独立**的。如果某一个过程满足马尔可夫性质，那么末来的转移与过去的是独立的，它只取决于现在。马尔可夫性质是 所有马尔可夫过程的基础。

## 1.2 马尔可夫链

马尔可夫过程是一组具有马尔可夫性质的随机变量序列 $s_1, \cdots, s_t$ ，其中下一个时刻的状态 $s_{t+1}$ 只取决于当前状态 $s_t$ 。我们设状态的历史为 $h_t=\left\{s_1, s_2, s_3, \ldots, s_t\right\}$ ( $h_t$ 包含了之前的所有状态)，则马尔可夫过程满足条件:

$$
p\left(s_{t+1} \mid s_t\right)=p\left(s_{t+1} \mid h_t\right)
$$

从当前 $s_t$ 转移到 $s_{t+1}$ ，它是直接就等于它之前所有的状态转移到 $s_{t+1}$ 。**离散时间的马尔可夫过程**也称为**马尔可夫链** (Markov chain) 。马尔可夫链是最简单的马尔可夫过程，其状态是有限的。

我们可以用状态转移矩阵 (state transition matrix) $\boldsymbol{P}$ 来描述状态转移 $p\left(s_{t+1}=s^{\prime} \mid s_t=s\right)$ :

$$
\boldsymbol{P}=\left(\begin{array}{cccc}
p\left(s_1 \mid s_1\right) & p\left(s_2 \mid s_1\right) & \ldots & p\left(s_N \mid s_1\right) \\
p\left(s_1 \mid s_2\right) & p\left(s_2 \mid s_2\right) & \ldots & p\left(s_N \mid s_2\right) \\
\vdots & \vdots & \ddots & \vdots \\
p\left(s_1 \mid s_N\right) & p\left(s_2 \mid s_N\right) & \ldots & p\left(s_N \mid s_N\right)
\end{array}\right)
$$

状态转移矩阵类似于条件概率 (conditional probability)，它表示当我们知道当前我们在状态 $s_t$ 时，到达下面所有状态的概率。所以它的每一行描述的是从一个节点到达所有其他节点的概率。

# 二、马尔可夫奖励过程

MRP定义 

- $S$ 有限状态集
- $P$ 转移模型，$P\left(s_{t+1}=s^{\prime} \mid s_t=s\right)$
- $R$ 奖励函数 $R\left(s_t=s, \right)=\mathbb{E}\left[r_t \mid s_t=s\right]$
- 折扣银子 $\gamma \in[0,1]$
- MDP 是一个元组: $(S, P, R, \gamma)$。

## 2.1 回报和价值函数

马尔可夫奖励过程（Markov reward process, MRP）是**马尔可夫链**加上**奖励函数**。在马尔可夫奖励过程中，状态转移矩阵和状态都与马尔可夫链一样，只是多了**奖励函数（reward function）**。奖励函数$R$是一个期望，表示当我们到达某一个状态的时候，可以获得多大的奖励。这里另外定义了折扣因子$\gamma$。如果状态数是有限的，那么$R$可以是一个向量。

1. 范围（horizon）：是指一个回合的长度（每个回合最大的时间步数），它是由有限个步数决定的。
2. 回报（return）：可以定义为奖励的逐步叠加，时刻$t$后的奖励序列为：
    
    $$
    G_t=r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\gamma^3 r_{t+4}+\ldots+\gamma^{T-t-1} r_T
    $$
    
    其中， $T$ 是最终时刻， $\gamma$ 是折扣因子，越往后得到的奖励，折扣越多。这说明我们更希望得到现有的奖励，对末来的奖励要打折扣。当我们有了回 报之后，就可以定义状态的价值了，就是状态价值函数（state-value function）。对于马尔可夫奖励过程，状态价值函数被定义成回报的期望，即
    
    $$
    \begin{aligned}
    V_t(s) & =\mathbb{E}\left[G_t \mid s_t=s\right] \\
    & =\mathbb{E}\left[r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\ldots+\gamma^{T-t-1} r_T \mid s_t=s\right]
    \end{aligned}
    $$
    
    其中， $G_t$ 是之前定义的折扣回报（discounted return) 。我们对 $G_t$ 取了一个期望，期望就是从这个状态开始，我们可能获得多大的价值。所以期望也可以看成末来可能获得奖励的当前价值的表现，就是当我们进入某一个状态后，我们现在有多大的价值。
    
    <aside>
    😍 为什么用折扣因子？
    
    **第一**，有些马尔可夫过程是带环的，它并不会终结，我们想避免无穷的奖励。
    
    **第二**，我们并不能建立完美的模拟环境的模型，我们对未来的评估不一定是准确的，我们不一定完全信任模型，因为这种不确定性，所以我们对未来的评估增加一个折扣。我们想把这个不确定性表示出来，希望尽可能快地得到奖励，而不是在未来某一个点得到奖励。
    
    **第三**，如果奖励是有实际价值的，我们可能更希望立刻就得到奖励，而不是后面再得到奖励（现在的钱比以后的钱更有价值）。
    
    **最后**，我们也更想得到即时奖励。有些时候可以把折扣因子设为 0，我们就只关注当前的奖励。我们也可以把折扣因子设为 1，，对未来的奖励并没有打折扣，未来获得的奖励与当前获得的奖励是一样的。折扣因子可以作为强化学习智能体的一个超参数来进行调整，通过调整折扣因子，我们可以得到不同动作的智能体。
    
    </aside>
    

## 2.2 贝尔曼方程

从价值函数里面推导出**贝尔曼方程（Bellman equation）**，先上结论：

$$
V(s)=\underbrace{R(s)}_{\text {即时奖励 }}+\underbrace{\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)}_{\text {末来奖励的折扣总和 }}
$$

其中，

- $s^{\prime}$ 可以看成末来的所有状态，
- $p\left(s^{\prime} \mid s\right)$ 是指从当前状态转移到末来状态的概率。
- $V\left(s^{\prime}\right)$ 代表的是末来某一个状态的价值。我们从当前状态开始，有一定的概率去到末来的所有状态，所以我们要把 $p\left(s^{\prime} \mid s\right)$ 写上去。我们得到了末来状态后，乘一个 $\gamma$ ，这样就可以把末来的奖励打折扣。
- $\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)$ 可以看成末来奖励的折扣总和 (discounted sum of future reward)。
贝尔曼方程定义了当前状态与末来状态之间的关系。末来奖励的折扣总和加上即时奖励，就组成了贝尔曼方程。
- 推导过程
    
    可以先证明：$\mathbb{E}\left[V\left(s_{t+1}\right) \mid s_t\right]=\mathbb{E}\left[\mathbb{E}\left[G_{t+1} \mid s_{t+1}\right] \mid s_t\right]=\mathbb{E}\left[G_{t+1} \mid s_t\right]$。令$s=s_t, g^{\prime}=G_{t+1}, s^{\prime}=s_{t+1}$
    
    $$
    \begin{aligned}\mathbb{E}\left[\mathbb{E}\left[G_{t+1} \mid s_{t+1}\right] \mid s_t\right] & =\mathbb{E}\left[\mathbb{E}\left[g^{\prime} \mid s^{\prime}\right] \mid s\right] \\& =\mathbb{E}\left[\sum_{g^{\prime}} g^{\prime} p\left(g^{\prime} \mid s^{\prime}\right) \mid s\right] \\& =\sum_{s^{\prime}} \sum_{g^{\prime}} g^{\prime} p\left(g^{\prime} \mid s^{\prime}, s\right) p\left(s^{\prime} \mid s\right) \\& =\sum_{s^{\prime}} \sum_{g^{\prime}} \frac{g^{\prime} p\left(g^{\prime} \mid s^{\prime}, s\right) p\left(s^{\prime} \mid s\right) p(s)}{p(s)} \\& =\sum_{s^{\prime}} \sum_{g^{\prime}} \frac{g^{\prime} p\left(g^{\prime} \mid s^{\prime}, s\right) p\left(s^{\prime}, s\right)}{p(s)} \\& =\sum_{s^{\prime}} \sum_{g^{\prime}} \frac{g^{\prime} p\left(g^{\prime}, s^{\prime}, s\right)}{p(s)} \\& =\sum_{s^{\prime}} \sum_{g^{\prime}} g^{\prime} p\left(g^{\prime}, s^{\prime} \mid s\right) \\& =\sum_{g^{\prime}} \sum_{s^{\prime}} g^{\prime} p\left(g^{\prime}, s^{\prime} \mid s\right) \\& =\sum_{g^{\prime}} g^{\prime} p\left(g^{\prime} \mid s\right) \\& =\mathbb{E}\left[g^{\prime} \mid s\right]=\mathbb{E}\left[G_{t+1} \mid s_t\right]\end{aligned}
    $$
    
    $$
    \begin{aligned}V(s) & =\mathbb{E}\left[G_t \mid s_t=s\right] \\& =\mathbb{E}\left[R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\ldots \mid s_t=s\right] \\& =\mathbb{E}\left[R_{t+1} \mid s_t=s\right]+\gamma \mathbb{E}\left[R_{t+2}+\gamma R_{t+3}+\gamma^2 R_{t+4}+\ldots \mid s_t=s\right] \\& =R(s)+\gamma \mathbb{E}\left[G_{t+1} \mid s_t=s\right] \\& =R(s)+\gamma \mathbb{E}\left[V\left(s_{t+1}\right) \mid s_t=s\right] \\& =R(s)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)\end{aligned}
    $$
    

可以把贝尔曼方程写成矩阵的形式：

$$
\left(\begin{array}{c}V\left(s_1\right) \\V\left(s_2\right) \\\vdots \\V\left(s_N\right)\end{array}\right)=\left(\begin{array}{c}R\left(s_1\right) \\R\left(s_2\right) \\\vdots \\R\left(s_N\right)\end{array}\right)+\gamma\left(\begin{array}{cccc}p\left(s_1 \mid s_1\right) & p\left(s_2 \mid s_1\right) & \ldots & p\left(s_N \mid s_1\right) \\p\left(s_1 \mid s_2\right) & p\left(s_2 \mid s_2\right) & \ldots & p\left(s_N \mid s_2\right) \\\vdots & \vdots & \ddots & \vdots \\p\left(s_1 \mid s_N\right) & p\left(s_2 \mid s_N\right) & \ldots & p\left(s_N \mid s_N\right)\end{array}\right)\left(\begin{array}{c}V\left(s_1\right) \\V\left(s_2\right) \\\vdots \\V\left(s_N\right)\end{array}\right)
$$

每一行来看，向量$V$乘状态转移矩阵里面的某一行，再加上它当前可以得到的奖励，就会得到它当前的价值。当我们把贝尔曼方程写成矩阵形式后，可以直接求解：

$$
\begin{aligned}& \boldsymbol{V}=\boldsymbol{R}+\gamma \boldsymbol{P} \boldsymbol{V}  \\& \boldsymbol{V}=(\boldsymbol{I}-\gamma \boldsymbol{P})^{-1} \boldsymbol{R} \end{aligned}
$$

但是求解复杂度是$O(N^3)$。求解算法有**蒙特卡洛**、**动态规划**和**时序差分学习**

![                                                                  蒙特卡洛算法](%E7%AC%AC%E4%BA%8C%E7%AB%A0%20Markov%20Decision%20Process%20(%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B)%20080bf490ade74ef18bcbe266cbe65200/Untitled.png)

                                                                  蒙特卡洛算法

![                                                                  动态规划算法](%E7%AC%AC%E4%BA%8C%E7%AB%A0%20Markov%20Decision%20Process%20(%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B)%20080bf490ade74ef18bcbe266cbe65200/Untitled%201.png)

                                                                  动态规划算法

# 三、马尔可夫决策过程

MDP定义 

- $S$ 有限状态集
- $A$ 有限动作集
- $P^a$ 动作转移模型，$P\left(s_{t+1}=s^{\prime} \mid s_t=s, a_t=a\right)$
- $R$ 奖励函数 $R\left(s_t=s, a_t=a\right)=\mathbb{E}\left[r_t \mid s_t=s, a_t=a\right]$
- 折扣因子 $\gamma \in[0,1]$
- MDP 是一个元组: $(S, A, P, R, \gamma)$。

## 3.1 策略

1. 策略定义了在某一个状态应该采取什么样的动作。知道当前状态后，我们可以把当前状态代入策略函数来得到一个概率，即
    
    $$
    \pi(a \mid s)=p\left(a_t=a \mid s_t=s\right)
    $$
    
    概率代表在所有可能的动作里面怎样采取行动，比如可能有 0.7 的概率往左走，有 0.3 的概率往右走，这是一个概率的表示。另外策略也可能是确定的，它有可能直接输出一个值，或者直接告诉我们当前应该采取什么样的动作，而不是一个动作的概率。假设概率函数是平稳的（stationary），不同时间点，我们采取的动作其实都是在对策略函数进行采样。
    
2. 已知马尔可夫决策过程和策略 $\pi$ ，我们可以把马尔可夫决策过程转换成马尔可夫奖励过程。在马尔可夫决策过程里面，状态转移函$P(s′∣s,a)$ 基于它当前的状态以及它当前的动作。因为我们现在已知策略函数，也就是已知在每一个状态下，可能采取的动作的概率，所以我们就可以直接把动作进行加和，marginalize $a$，这样我们就可以得到对于马尔可夫奖励过程的转移，这里就没有动作，
    
    $$
    P_\pi\left(s^{\prime} \mid s\right)=\sum_{a \in A} \pi(a \mid s) p\left(s^{\prime} \mid s, a\right)
    $$
    
    对于奖励函数，我们也可以把动作去掉，这样就会得到类似于马尔可夫奖励过程的奖励函数，即
    
    $$
    r_\pi(s)=\sum_{a \in A} \pi(a \mid s) R(s, a)
    $$
    

## 3.2 同过程、奖励过程的区别

![Untitled](%E7%AC%AC%E4%BA%8C%E7%AB%A0%20Markov%20Decision%20Process%20(%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B)%20080bf490ade74ef18bcbe266cbe65200/Untitled%202.png)

1. 马尔可夫过程/马尔可夫奖励过程的状态转移是直接决定的。比如当前状态是$s$，那么直接通过转移概率决定下一个状态是什么。
2. 但对于马尔可夫决策过程，它的中间多了一层动作$a$, 即智能体在当前状态的时候，首先要决定采取某一种动作，这样我们会到达某一个黑色的节点。到达这个黑色的节点后，因为有一定的不确定性，所以当智能体当前状态以及智能体当前采取的动作决定过后，智能体进入未来的状态其实也是一个概率分布。在当前状态与未来状态转移过程中多了一层决策性，这是马尔可夫决策过程与之前的马尔可夫过程/马尔可夫奖励过程很不同的一点。在马尔可夫决策过程中，动作是由智能体决定的，智能体会采取动作来决定未来的状态转移。

## 3.3 价值函数

马尔可夫决策过程中的价值函数可定义为

$$
V_\pi(s)=\mathbb{E}_\pi\left[G_t \mid s_t=s\right]
$$

其中，期望基于我们采取的策略。当策略决定后，我们通过对策略进行采样来得到一个期望，计算出它的价值函数。这里我们另外引入了一个 **Q 函数（Q-function）**。Q 函数也被称为**动作价值函数（action-value function）**。Q 函数定义的是在某一个状态采取某一个动作，它有可能得到的回报的一个期望，即

$$
Q_\pi(s, a)=\mathbb{E}_\pi\left[G_t \mid s_t=s, a_t=a\right]
$$

这里的期望其实也是基于策略函数的。所以我们需要对策略函数进行一个加和，然后得到它的价值。 对 Q 函数中的动作进行加和，就可以得到价值函数：

$$
V_\pi(s)=\sum_{a \in A} \pi(a \mid s) Q_\pi(s, a)
$$

## 3.4 贝尔曼期望方程

我们可以把状态价值函数和 Q 函数拆解成两个部分：即时奖励和后续状态的折扣价值（discounted value of successor state）。 通过对价值函数进行分解，我们就可以得到一个类似于之前马尔可夫奖励过程的贝尔曼方程————**贝尔曼期望方程（Bellman expectation equation）**

$$
V_\pi(s)=\mathbb{E}_\pi\left[r_{t+1}+\gamma V_\pi\left(s_{t+1}\right) \mid s_t=s\right]
$$

$$
V_\pi(s)=\sum_{a \in A} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V_\pi\left(s^{\prime}\right)\right)
$$

- 推导：
    
    $$
    \begin{aligned}Q(s, a) & =\mathbb{E}\left[G_t \mid s_t=s, a_t=a\right] \\& =\mathbb{E}\left[r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\ldots \mid s_t=s, a_t=a\right] \\& =\mathbb{E}\left[r_{t+1} \mid s_t=s, a_t=a\right]+\gamma \mathbb{E}\left[r_{t+2}+\gamma r_{t+3}+\gamma^2 r_{t+4}+\ldots \mid s_t=s, a_t=a\right] \\& =R(s, a)+\gamma \mathbb{E}\left[G_{t+1} \mid s_t=s, a_t=a\right] \\& =R(s, a)+\gamma \mathbb{E}\left[V\left(s_{t+1}\right) \mid s_t=s, a_t=a\right] \\& =R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V\left(s^{\prime}\right)\end{aligned}
    $$
    

对于 Q 函数，我们也可以做类似的分解，得到 Q 函数的贝尔曼期望方程：

$$
Q_\pi(s, a)=\mathbb{E}_\pi\left[r_{t+1}+\gamma Q_\pi\left(s_{t+1}, a_{t+1}\right) \mid s_t=s, a_t=a\right]
$$

$$
Q_\pi(s, a)=R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) \sum_{a^{\prime} \in A} \pi\left(a^{\prime} \mid s^{\prime}\right) Q_\pi\left(s^{\prime}, a^{\prime}\right)\\=R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right)\textcolor{orange}{V_\pi (s')}
$$

<aside>
💡 这里可以通过V函数求出Q函数！

</aside>

## 3.5 备份图

对于某一个状态，它的当前价值是与它的未来价值线性相关的。 我们称为**备份图（backup diagram）**或回溯图，因为它们所示的关系构成了更新或备份操作的基础，而这些操作是强化学习方法的核心。这些操作将价值信息从一个状态（或状态-动作对）的后继状态（或状态-动作对）转移回它。 每一个空心圆圈代表一个状态，每一个实心圆圈代表一个状态-动作对。 

1. 对于V-function：
    
    $$
    v_\pi(s)=\sum_{a \in A} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s, a\right) v_\pi\left(s^{\prime}\right)\right)
    $$
    
    ![Untitled](%E7%AC%AC%E4%BA%8C%E7%AB%A0%20Markov%20Decision%20Process%20(%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B)%20080bf490ade74ef18bcbe266cbe65200/Untitled%203.png)
    
2. 对于Q-function
    
    $$
    q_\pi(s, a)=R(s, a)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s, a\right) \sum_{a^{\prime} \in A} \pi\left(a^{\prime} \mid s^{\prime}\right) q_\pi\left(s^{\prime}, a^{\prime}\right)
    $$
    
    ![Untitled](%E7%AC%AC%E4%BA%8C%E7%AB%A0%20Markov%20Decision%20Process%20(%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B)%20080bf490ade74ef18bcbe266cbe65200/Untitled%204.png)
    

## 3.6 策略评估

已知马尔可夫决策过程以及要采取的策略$\pi$，计算价值函数$V_\pi(s)$的过程就是**策略评估（价值预测）**。也就是预测我们当前采取的策略最终会产生多少价值。

我们可以直接通过贝尔曼期望方程来得到价值函数：

$$
V^\pi_t(s)=R_\pi(s)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, \pi(s)\right) V^\pi_{t-1}\left(s^{\prime}\right)
$$

我们可以不停用贝尔曼期望方程迭代，最后价值函数会收敛。收敛之后，价值函数的值就是每一个状态的价值。

## 3.7 预测和控制

预测 (prediction) 和控制 (control) 是马尔可夫决策过程里面的核心问题。

1. **预测**（评估一个给定的策略) : 预测是指给定一个马尔可夫决策过程以及一个策略 $\pi$ ，计算它的**价值函数**，也就是**计算每个状态的价值**。
    1. 输入是马尔可夫决策过程 $<S, A, P, R, \gamma>$ 和策略 $\pi$ ，
    2. 输出是价值函数 $V_\pi$ 。
2. **控制** (搜索最佳策略) : 控制就是我们去**寻找一个最佳的策略**，然后同时**输出它的最佳价值函数**以及最**佳策略**。
    1. 输入是马尔可夫决策过程 $<S, A, P, R, \gamma>$ ，
    2. 输出是**最佳价值函数** (optimal value function) $V^*$和**最佳策略** (optimal policy） *$\pi^*$* 。

<aside>
😍 在马尔可夫决策过程里面，预测和控制都可以通过动态规划解决。要强调的是，这两者的区别就在于，预测问题是给定一个策略，我们要确定它的价值函数是多少。而控制问题是在没有策略的前提下，我们要确定最佳的价值函数以及对应的决策方案。实际上，这两者是递进的关系，在强化学习中，我们**通过解决预测问题，进而解决控制问题**。

</aside>

## 3.8 预测（策略评估）

策略评估就是给定马尔可夫决策过程和策略，评估我们可以获得多少价值，即对于当前策略，我们可以得到多大的价值。我们可以直接把**贝尔曼期望备份（Bellman expectation backup）** ，变成迭代的过程，反复迭代直到收敛。这个迭代过程可以看作**同步备份（synchronous backup）** 的过程。

$$
V_{t+1}(s)=\sum_{a \in A} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V_t\left(s^{\prime}\right)\right)
$$

策略评估的核心思想就是把如式所示的贝尔曼期望备份反复迭代，然后得到一个收敛的价值函数的值。因为已经给定了**策略函数**，所以我们可以直接把它简化成一个**马尔可夫奖励过程**的表达形式，相当于把$a$去掉，即

$$
V_{t+1}(s)=r_\pi(s)+\gamma p_\pi\left(s^{\prime} \mid s\right) V_t\left(s^{\prime}\right)
$$

## 3.9 控制

策略评估是指给定马尔可夫决策过程和策略，我们可以估算出价值函数的值。如果我们只有马尔可夫决策过程，那么应该如何寻找最佳的策略，从而得到**最佳价值函数（optimal value function）**呢？

最佳价值函数的定义为：

$$
V^*(s)=\max _\pi V_\pi(s)
$$

$$
\pi^*(s)=\underset{\pi}{\arg \max } V_\pi(s)
$$

当取得最佳价值函数后，我们可以通过对 $Q$ 函数进行最大化来得到最佳策略（这里是deterministic的！）：

$$
\pi^*(a \mid s)= \begin{cases}1, & a=\underset{a \in A}{\arg \max } Q^*(s, a) \\ 0, & \text { 其他 }\end{cases}
$$

当$Q$函数收敛后，因为$Q$函数是关于状态与动作的函数，所以如果在某个状态采取某个动作，可以使得$Q$函数最大化，那么这个动作就是最佳的动作。如果我们能优化出一个 $Q$ 函数$Q^*(s,a)$，就可以直接在$Q$函数中取一个让$Q$函数值最大化的动作的值，就可以提取出最佳策略。

我们可以通过策略迭代和价值迭代来解决马尔可夫决策过程的控制问题。

## 3.10 策略迭代

1. 策略迭代由两个步骤组成：**策略评估**和**策略改进**（policy improvement）。
    1. 策略评估：当前我们在优化策略$\pi$，在优化过程中得到一个最新的策略。我们先保证这个策略不变，然后估计它的价值，即给定当前的策略函数来估计状态价值函数。 通过**贝尔曼期望方程**迭代，得到价值函数$V_{\pi_i}$。
    2. 策略改进：得到价值函数后，我们可以进一步推算出它的 $Q$ 函数。得到 $Q$ 函数后，我们直接对 $Q$ 函数进行最大化，通过在 $Q$ 函数做一个贪心的搜索来进一步改进策略。这两个步骤一直在迭代进行。在策略迭代里面，在初始化的时候，我们有一个初始化的状态价值函数$V$和策略$\pi$，然后在这两个步骤之间迭代。
        
        $$
        Q_{\pi_i}(s, a)=R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V_{\pi_i}\left(s^{\prime}\right)
        $$
        
        对于每个状态，策略改进会得到它的新一轮的策略，对于每个状态，我们取使它得到最大值的动作，即
        
        $$
        \pi_{i+1}(s)=\underset{a}{\arg \max } Q_{\pi_i}(s, a)
        $$
        
    
    ![Untitled](%E7%AC%AC%E4%BA%8C%E7%AB%A0%20Markov%20Decision%20Process%20(%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B)%20080bf490ade74ef18bcbe266cbe65200/Untitled%205.png)
    
    上面的线就是我们当前状态价值函数的值，下面的线是策略的值。 策略迭代的过程与踢皮球一样。我们先给定当前已有的策略函数，计算它的状态价值函数。算出状态价值函数后，我们会得到一个 Q 函数。我们对Q 函数采取贪心的策略，这样就像踢皮球，“踢”回策略。然后进一步改进策略，得到一个改进的策略后，它还不是最佳的策略，我们再进行策略评估，又会得到一个新的价值函数。基于这个新的价值函数再进行 Q 函数的最大化，这样逐渐迭代，状态价值函数和策略就会收敛。
    
2. 贝尔曼最优方程: 
当我们一直采取 argmax 操作的时候，我们会得到一个单调的递增。通过采取这种贪心操作（argmax 操作），我们就会得到更好的或者不变的策略，而不会使价值函数变差。所以当改进停止后，我们就会得到一个最佳策略。当改进停止后，我们取让 Q 函数值最大化的动作，Q 函数就会直接变成价值函数，即
    
    $$
    Q_\pi\left(s, \pi^{\prime}(s)\right)=\max _{a \in A} Q_\pi(s, a)=Q_\pi(s, \pi(s))=V_\pi(s)
    $$
    
    $$
    V_\pi(s)=\max _{a \in A} Q_\pi(s, a)
    $$
    
    <aside>
    💡 这里因为$V_\pi(s)=\sum_{a \in A} \pi(a \mid s) Q_\pi(s, a)$，而$\pi(a \mid s)$是只有$a=\pi^*(s)$为1
    
    </aside>
    
    贝尔曼最优方程表明：最佳策略下的一个状态的价值必须等于在这个状态下采取最好动作得到的回报的期望。 当马尔可夫决策过程满足贝尔曼最优方程的时候，整个马尔可夫决策过程已经达到最佳的状态。
    
    只有当整个状态已经收敛后，我们得到最佳价值函数后，贝尔曼最优方程才会满足。满足贝尔曼最优方程后，我们可以采用最大化操作，即
    
    $$
    \begin{aligned}
    Q^*(s, a) & =R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V^*\left(s^{\prime}\right) \\
    & =R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) \max _a Q^*\left(s^{\prime}, a^{\prime}\right)
    \end{aligned}
    $$
    

## 3.11 价值迭代

1. 最优性原理
    
    我们从另一个角度思考问题，动态规划的方法将优化问题分成两个部分。第一步执行的是最优的动作。之后后继的状态的每一步都按照最优的策略去做，最后的结果就是最优的。
    
2. 最优性原理定理（principle of optimality theorem）： 一个策略$π(a∣s)$ 在状态 $s$ 达到了最优价值，也就是 $V^π(s)=V^∗(s)$ 成立，当且仅当对于任何能够从 $s$ 到达的 $s'$，都已经达到了最优价值。也就是对于所有的$s'$，$V^π(s')=V^∗(s')$  恒成立。
3. 确认性价值迭代
    
    如果我们知道子问题 $V^∗(s')$ 的最优解，就可以通过价值迭代来得到最优的$V^∗(s)$的解。价值迭代就是把贝尔曼最优方程当成一个更新规则来进行，即
    
    $$
    V(s) \leftarrow \max _{a \in A}\left(R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V\left(s^{\prime}\right)\right)
    $$
    
    只有当整个马尔可夫决策过程已经达到最佳的状态时，式才满足。但我们可以把它转换成一个备份的等式。备份的等式就是一个迭代的等式。我们不停地迭代贝尔曼最优方程，价值函数就能逐渐趋向于最佳的价值函数，这是价值迭代算法的精髓。
    
    为了得到最佳的$V^*$ ，对于每个状态的 $V$，我们直接通过贝尔曼最优方程进行迭代，迭代多次之后，价值函数就会收敛。这种价值迭代算法也被称为**确认性价值迭代**（deterministic value iteration）。
    
4. 具体算法：价值迭代算法的过程如下。
    - 初始化: 令 $k=1$，对于所有状态 $s ， V_0(s)=0$ 。
    - 对于 $k=1: H$ ( $H$ 是让 $V(s)$ 收敛所需的迭代次数)
        - 对于所有状态 $s$
            
            $$
            \begin{gathered}
            Q_{k+1}(s, a)=R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V_k\left(s^{\prime}\right) \\
            V_{k+1}(s)=\max_a Q_{k+1}(s, a)
            \end{gathered}
            $$
            
        - $k \leftarrow k+1$
    - 在迭代后提取最优策略:
        
        $$
        \pi(s)=\underset{a}{\arg \max }\left[R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V_{H+1}\left(s^{\prime}\right)\right]
        $$
        

## 3.12 对比策略迭代和价值迭代

这两个算法都可以解马尔可夫决策过程的控制问题。

1. 策略迭代分两步。
    1. 首先进行策略评估，即对当前已经搜索到的策略函数进行估值，通过迭代计算出价值函数$V$
    2. 得到估值后，我们进行策略改进，即把 $Q$ 函数算出来，进行进一步改进。不断重复这两步，直到策略收敛。
2. 价值迭代直接使用**贝尔曼最优方程**进行迭代，从而寻找最佳的价值函数。找到最佳价值函数后，我们再提取一次最佳策略（一旦价值函数是最优的，策略也是最优的）。

## 3.13 总结

| 问题 | 贝尔曼方程 | 算法 |
| --- | --- | --- |
| 预测 | 贝尔曼期望方程 | 策略评估 |
| 控制 | 贝尔曼期望方程 | 策略迭代 |
| 控制 | 贝尔曼最优方程 | 价值迭代 |