# Comprehensive Guide: KTH IPP Traffic Generation, Model Architectures, and Evaluation Metrics

This document provides a deeply detailed, mathematically rigorous explanation of the entire forecasting and data generation pipeline. It covers the four machine learning forecasting models evaluated, the specific loss functions used to measure them, and a full theoretical and numerical breakdown of the KTH Interacting Particle Process (IPP) used to synthetically inflate our telecom datasets.

---

## 1. The Four Forecasting Models

To adequately capture both point forecasting (single expected values) and probabilistic forecasting (uncertainty estimations) of the highly volatile 5G network traffic, four distinct models were selected.

### 1.1 SARIMA (Seasonal Auto-Regressive Integrated Moving Average)
**Type:** Classical Statistical Model  
**Explanation:** SARIMA establishes a reliable linear baseline. It models the temporal dependencies in the traffic by accounting for auto-regression (past values influence future values), differencing (stationarizing trends), and moving averages (past forecast errors influence future forecasts), alongside explicit seasonal components. It is excellent at safely forecasting moving averages but generally struggles to capture complex non-linear burstiness without smoothing it out into a straight trajectory.

### 1.2 LSTM (Long Short-Term Memory Network)
**Type:** Deep Learning Point-Forecaster  
**Explanation:** LSTM is a type of Recurrent Neural Network (RNN) structured specifically to circumvent the vanishing gradient problem in sequential data. Using its sophisticated gating mechanisms (Forget, Input, Output gates), an LSTM can selectively remember or forget long-term dependencies within the traffic history sequence. When trained accurately with sufficient window dimensions, the LSTM is theoretically the most powerful architecture for extracting deterministic, non-linear point estimates from the traffic series.

### 1.3 Chronos V1 & Chronos 2
**Type:** Foundation Time-Series Models (Zero-Shot)  
**Explanation:** The Chronos architecture (by Amazon) conceptualizes time-series forecasting equivalently to Natural Language Processing language models. It tokenizes the numerical traffic data into a discrete vocabulary and parses it through a Transformer architecture. 
- **Chronos V1:** Utilizes a standard language modeling approach heavily inspired by T5, generating probabilistic future sequences token by token. It attempts to physically simulate the probabilistic distribution of future paths (which causes it to actively try and model the "bursty" visual patterns).
- **Chronos 2:** A newer iteration built with structural optimizations. It significantly improves upon the statistical calibration of zero-shot forecasting, capturing incredibly accurate median trajectories natively without fine-tuning, thus achieving industry-leading RMSE on uncertain sequences.

---

## 2. Evaluation Metrics: RMSE vs. Pinball Loss

A dual-metric approach was selected to provide a holistic evaluation of the models, separating their ability to guess the exact numerical average from their ability to estimate boundary risks.

### 2.1 RMSE (Root Mean Squared Error)
**Definition:** $\text{RMSE} = \sqrt{\frac{1}{n} \Sigma_{i=1}^{n} (y_i - \hat{y}_i)^2}$
**Why it's used:** RMSE penalizes larger errors exponentially more than smaller errors due to squaring. In traffic forecasting, giant bursts or sudden drop-offs hurt the network heavily, so predicting those (or safely staying in the middle) minimizes severe structural penalties. RMSE is the gold standard for measuring how close a single **point-forecast** is to the actual reported number overall.

### 2.2 Pinball Loss (Quantile Loss)
**Definition for quantile $q$:** 
$L_q(y, \hat{y}) = (y - \hat{y}) \cdot q \quad \text{if } y \ge \hat{y}$
$L_q(y, \hat{y}) = (\hat{y} - y) \cdot (1 - q) \quad \text{if } \hat{y} > y$

**Why it's used:** Unlike RMSE, which cares about the "mean" (center), Pinball Loss evaluates **probabilistic bounds**. In telecommunications, predicting the mean isn't always enough—you need to know the *worst-case scenarios* to provision enough bandwidth (e.g., evaluating the 90th or 95th percentile limits). Pinball Loss strictly scores a model based on how accurately its upper-bound projections capture those percentile margins without overestimating unnecessarily. 

---

## 3. The KTH Interacting Particle Process (IPP) Synthesis

The core of our augmentation strategy relies on mapping coarse-grained aggregated samples (e.g., 5-minute averages) to ultra-fine high-resolution structures while strictly preserving their mathematical moments (Mean and Variance).

### 3.1 The Continuous-Time Markov Chain (CTMC)
The model assumes that underlying network activity switches between active bursts (ON) and silence (OFF).
- **$\tau$ (tau):** The rate of transitioning from OFF to ON. (Time spent OFF ~ $Exp(\tau)$).
- **$\zeta$ (zeta):** The rate of transitioning from ON to OFF. (Time spent ON ~ $Exp(\zeta)$).

The stationary probability of being in the ON state is defined as:
$$P_{on} = \frac{\tau}{\tau + \zeta}$$

During ON periods, traffic sessions arrive following a Poisson process with rate $\lambda$. Each session transmits an amount of data drawn from an Exponential distribution with mean $\psi_{mean}$.

### 3.2 Moment Matching Equations
Our objective is to solve for the microscopic parameters ($\lambda$ and $\psi_{mean}$) for every slot of duration $T$, using the empirical Mean ($E_{rate}$) and empirical Variance ($E_{var}$).

**The Theoretical Equations:**
1) Mean constraint:
$$E_{rate} = P_{on} \cdot \lambda \cdot \psi_{mean}$$

2) Variance constraint:
$$E_{var} = \frac{P_{on} \cdot \lambda \cdot 2 \psi_{mean}^2}{T} + \left( \frac{2\zeta}{\tau T (\tau + \zeta)} \right) \cdot E_{rate}^2$$

### 3.3 The Rolling Variance Proxy
Because we only possess a single point of data per 5-minute coarse slot, we cannot calculate instantaneous variance directly. Instead, we use a **Rolling Variance Proxy $v$**.

1. Compute empirical rolling variance $v$ over window $k=6$.
2. **Generic Floor:** To prevent $v$ from becoming dangerously close to 0 in inactive slots, we securely bound it proportional to the rate:
$$Floor_{generic} = (Ratio \cdot E_{rate})^2$$
where Ratio is $0.01$. Thus, $v \leftarrow \max(v, Floor_{generic})$.

3. **KTH Feasibility Floor:** This is mathematically critical. The variance equation contains a base variance intrinsic merely to the CTMC toggling ON and OFF, independent of the session sizes. If $v$ falls below this base variance, no real positive positive $\psi_{mean}$ can solve the equation.
$$Floor_{KTH} = \left( \frac{2\zeta}{\tau T (\tau + \zeta)} \right) \cdot E_{rate}^2 + \epsilon$$
We enforce: $v \leftarrow \max(v, Floor_{KTH})$. Finally, we apply a synthetic `variance_scale` multiplier to naturally curb synthesized volatility while respecting equations boundaries.

### 3.4 Numerical Example Execution

Let's trace a concrete mathematical execution through the Python IPP codebase parameters.
Assume for a specific slot:
- $T = 300$ seconds, $\tau = 0.067$, $\zeta = 0.067$
- Empirical Coarse Rate $E_{rate} = 10.0$ Mbps
- Extracted Rolling Variance Proxy $E_{var} = 5.0$ Mbps$^2$

**Step 1: Calculate $P_{on}$ and CTMC Base Variance**
$$P_{on} = \frac{0.067}{0.067 + 0.067} = 0.5$$
$$Var_{CTMC\_Base\_Multiplier} = \frac{2 \cdot 0.067}{0.067 \cdot 300 \cdot (0.067 + 0.067)} = \frac{0.134}{2.6934} \approx 0.04975$$

**Step 2: Check Feasibility Floors**
The absolute mathematically lowest variance this IPP process can inherently produce at $E_{rate}=10.0$ is:
$$Floor_{KTH} = 0.04975 \cdot (10.0)^2 = 0.04975 \cdot 100 = 4.975 \text{ Mbps}^2$$
Since our extracted proxy $E_{var} = 5.0$ is slightly greater than $4.975$, the equation passes physical feasibility limits.

**Step 3: Solve for $\psi_{mean}$**
Subtract the CTMC base variance to find the required "Local Variance" ($V_{diff}$) that the actual sessions must supply to match empirical constraints:
$$V_{diff} = E_{var} - Floor_{KTH} = 5.0 - 4.975 = 0.025 \text{ Mbps}^2$$

Using the variance formula reformatted computationally:
$$\psi_{mean} = \frac{V_{diff} \cdot T}{2 \cdot E_{rate}} = \frac{0.025 \cdot 300}{2 \cdot 10.0} = \frac{7.5}{20.0} = 0.375 \text{ Mbits}$$

**Step 4: Solve for $\lambda$**
Using the Mean constraint computational inversion:
$$10.0 = P_{on} \cdot \lambda \cdot \psi_{mean} \implies 10.0 = 0.5 \cdot \lambda \cdot 0.375$$
$$\lambda = \frac{10.0}{0.1875} = 53.33 \text{ arrivals/second}$$

**Step 5: Safety Clamp Adjustments**
The code specifies a memory-protection `max_lam = 50.0`. 
Since $53.33 > 50.0$, we clamp $\lambda \leftarrow 50.0$.
To strictly preserve the $E_{rate} = 10.0$ volume integrity across simulation, $\psi_{mean}$ must recalibrate natively against the clamped rate:
$$\psi_{mean} = \frac{10.0}{0.5 \cdot 50.0} = \frac{10.0}{25.0} = 0.40 \text{ Mbits}$$

**Outcome:** The simulator will generate exactly 50 Poisson arrivals per second during ON periods, each transmitting rapidly drawn 0.40 Megabits of data chunks, thereby accurately echoing the target $10.0$ Mbps average constraint while perfectly distributing the ultra-fine local variance spikes calculated at $\approx 5.0$ Mbps$^2$.
