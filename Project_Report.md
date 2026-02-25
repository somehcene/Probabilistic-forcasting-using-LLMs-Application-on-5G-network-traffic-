# Project Report: KTH IPP-Based Traffic Augmentation

## 1. Overview
This report provides a detailed breakdown of the `src` directory containing the Continuous-Time Markov Chain (CTMC) Interrupted Poisson Process (IPP) traffic generator. The goal of this code is to take coarse, low-resolution network traffic data (e.g., 5-minute intervals) and generate realistic, highly-bursty fine-grained network traffic (e.g., 5-second intervals) that mathematically preserves the macroscopic volume of the original data.

---

## 2. Source Files & Function Breakdown

### `src/config.py`
Defines the structure and default settings for the augmentation process.
* **`KTHParams`**: A frozen dataclass storing all hyperparameters for the mathematical model:
  * `T=300`: The coarse slot duration in seconds (5 minutes).
  * `dt=5`: The fine-grained step duration in seconds.
  * `tau` & `zeta`: Transition rates for the ON/OFF states of the underlying CTMC.
  * Safety thresholds (`lambda_fixed`, `lambda_min`, `psi_mean_min`) to prevent division by zero or overly sparse data.
  * `session_mean_s`: Target average duration for a burst session.
* **`AugmentationReport`**: A dataclass strictly for keeping track of the generator’s tracking metrics (slots processed, infeasible states).

### `src/utils.py`
Utility functions supporting the statistical calculations needed before and after generation.
* **`rolling_variance_proxy`**: Calculates the local variance of the coarse data over a rolling window. It bounds this variance using generic floors and KTH-specific feasibility conditions to guarantee the math works when solving for expected IPP variance later.
* **`coarsen_fine_to_slots`**: Aggregates the generated fine-grained samples back to the coarse resolution by taking the mean. This is crucial for verifying that the exact volume of traffic was maintained.
* **`summary_errors`**: Calculates MAE, RMSE, and MAPE to statistically compare original vs. simulated trends.

### `src/kth_ipp.py`
The core mathematical engine of the generator.
* **`stationary_p_on(tau, zeta)`**: Computes the steady-state probability that the CTMC is in the "ON" state (`tau / (tau + zeta)`).
* **`solve_lambda_and_psi_mean(E_rate, E_var, p)`**: Takes the target empirical mean (`E_rate`) and variance (`E_var`) and solves a system of equations for two IPP parameters: the arrival rate $\lambda$ and the mean session size $\psi_{mean}$. It applies the KTH formulas for second-moment matching.
* **`simulate_ipp_slot(slot_params, p, rng)`**: The event-driven engine representing a single coarse slot:
  1. Cycles between ON/OFF intervals drawn from exponential distributions.
  2. Generates distinct session arrivals via a Poisson process during the ON time.
  3. Draws data volume (Mbits) for each arrival from an exponential distribution.
  4. Spreads the volume evenly over the session’s duration across small `dt` bins.
* **`generate_fine_series_from_coarse`**: The main interface. Iterates through the original traffic data, computing parameters via `solve_lambda_and_psi_mean` and running `simulate_ipp_slot`. Critically, it features a post-simulation re-scaling correction to mathematically adhere to the macroscopic mean.

---

## 3. Heuristics and Safety Measures Explained

Mathematical models running purely on probability can encounter pathological edge-cases. The code introduces several vital heuristics:

* **Constant Session Duration.** In `simulate_ipp_slot`, sessions are assigned a constant duration (`session_mean_s`) instead of a random exponential one. 
  * *Reason:* If a random duration is arbitrarily near zero while its assigned data amount is huge (drawn from an exponential distribution), it creates infinite Pareto-like spikes that completely destroy the simulation and explode the variance.
* **Minimum Lambda Enforcement.** In `solve_lambda_and_psi_mean`, if $\lambda$ drops below `p.lambda_fixed = 0.5`, it is clamped to `0.5`, and $\psi_{mean}$ is re-adjusted downward.
  * *Reason:* If $\lambda$ is infinitesimally small, the simulation might produce exactly *zero* arrivals for multiple consecutive slots, ignoring the target traffic. Re-adjusting $\psi_{mean}$ ensures the volume math stays conserved.
* **Truncation Over-Spill.** At the end of a slot, if a traffic session spills over the `T` boundary, the generator doesn't artificially compress the data into the remaining time. It allows the data to "spill out" and only logs the correct bit rate for the active bins.
* **Variance Floor.** `rolling_variance_proxy` prevents variance from dropping below a hard theoretical bound determined by the CTMC properties.
  * *Reason:* If empirical variance is too low, solving for $\psi_{mean}$ leads to a negative value. A hard floor prevents negative square roots.

---

## 4. The Past Problem: The "Aggregation Issue"

**What the problem was:** 
Previously, the generated fine-grained traffic appeared visibly lower than the original data's mean, or failed to perfectly aggregate back to the teacher's original 5-minute samples.

**Why it happened:**
An IPP is fundamentally stochastic (random). Even if `solve_lambda_and_psi_mean` calculates the exact mathematical parameters to approximate the average rate, running a random simulation for a brief 300-second window will inherently yield higher or lower volumes due purely to statistical variance. Sometimes a large burst triggers just before the 300 seconds are up—so it gets truncated, losing volume. Sometimes no burst happens at all.

**How it was fixed (Heuristic Post-Scaling):**
At the end of `generate_fine_series_from_coarse`, the simulator looks at the mean of the newly generated traffic (`m`) and compares it to the original target mean (`target`). It then applies a hard scaling factor:
`slot_rate *= (target / m)` 
This simple yet highly effective heuristic ensures perfect conservation of data volume. The micro-bursts and structural variance from IPP are preserved, but the total volume generated in the 300s window is strictly tethered to the original dataset.

---

## 5. Implications for Future Work (Forecasting with Chronos2 and LSTM)

This robust data-generation pipeline paves a crucial runway for your future AI models (Chronos2 and LSTMs) applied to 6G networks:

* **Training on Micro-Burstiness:** Real 5G/6G traffic is highly volatile at sub-second levels, driving latency and buffering events. A standard 5-minute dataset smoothes these spikes out completely. By expanding the dataset with realistic IPP micro-bursts, your LSTM/Chronos2 models are now forced to learn and predict rapid transients and volatility boundaries, not just sluggish diurnal waves.
* **Perfect Macro Trend Preservation:** Because the IPP generator strictly conserves the aggregate volume (due to the post-scaling step), the large-scale patterns of your network (peak hours vs. night hours) remain untouched. Zero-shot Time Series Foundation Models (like Chronos2) excel at these overarching trends, and your generated dataset will not corrupt that macroscopic signal.
* **Robust Stress-Testing:** Having fine-grained resolutions enables your models to be tested in "stress-test" scenarios (e.g., slicing allocation latency). Models will have to predict exactly when a micro-burst might saturate a specific network slice, making your final project evaluation highly aligned with advanced 6G QoS/QoE metrics.
