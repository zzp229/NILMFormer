<h1 align="center">NILMFormer</h1>

<p align="center">
    <img width="300" src="https://github.com/adrienpetralia/NILMFormer/blob/main/ressources/intro.png" alt="Intro image">
</p>

<h2 align="center">A Sequence-To-Sequence Non-Stationarity Aware Transformer for Non-Intrusive Load Monitoring</h2>


Millions of smart meters have been deployed worldwide, collecting the power consumed by individual households. Based on these measurements, electricity suppliers provide feedback on consumption behaviors.  
To help customers better understand their usage, suppliers need to provide **detailed** (per-appliance) feedback—a challenging problem known as **Non-Intrusive Load Monitoring (NILM)**.

NILM aims to disaggregate a household’s total power consumption and retrieve the individual power usage of different appliances. Current state-of-the-art (SotA) solutions rely on deep learning and process household consumption in subsequences. However, real-world smart meter data is **non-stationary**—distribution drifts within each window segment can severely impact model performance.

We introduce **NILMFormer**, a sequence-to-sequence Transformer-based architecture designed to tackle this problem.


## Outline 📝

This repository contains the **source code** of NILMFormer, as well as the code needed to reproduce the experimental evaluation from our paper.  
It also includes **10 recent SotA NILM baselines** re-implemented in PyTorch.

---

### Getting Started 🚀

To install the dependencies, you can use the following commands. Life is much easier thanks to [uv](https://astral.sh/blog/uv)!

```bash
pip install uv
git clone https://github.com/adrienpetralia/NILMFormer
cd NILMFormer
uv sync
```

### Launch an Experiment ⚙️

To run a *specific* experiment, use the command below:
```
uv run -m expes.launch_one_expe \
    --dataset "UKDALE" \
    --sampling_rate "1min" \
    --appliance "WashingMachine" \
    --window_size 128 \
    --name_model NILMFormer \
    --seed 0
```

To run all experiments conducted in our paper (this may take some time), use:
```
. expes/run_all_expes.sh
```

## NILMFormer Architecture

**TL;DR** : **NILMFormer** is a **sequence-to-sequence Transformer-based architecture** purpose-built for **Non-Intrusive Load Monitoring (NILM)**. It tackles the **non-stationary** nature of smart meter data by splitting and separately encoding the **shape**, **temporal** dynamics, and **intrinsic statistics** of each subsequence. These components are then fused within the Transformer block. Finally, the prediction is refined through a **linear transformation** of the input series statistics, accounting for **power loss** in the disaggregation process.


### Architecture Details

<p align="center">
    <img width="300" src="https://github.com/adrienpetralia/NILMFormer/blob/main/ressources/nilmformer.png" alt="NILMFormer Architecture">
</p>

To handle the non-stationarity aspect of electricity consumption data, NILMFormer operates by first stationnarizing the input subsequence by subtracting its mean and standard deviation.
While the normalized subsequence is passed through a robust convolutional block that serves as a features extractor, the removed statistics are linearly projected in a higher space (referred to as *TokenStats*), and the timestamps are used by the proposed TimeRPE module to compute a positional encoding matrix.
These features are concatenated and fed into the Transformer block, followed by a simple Head to obtain a 1D sequence of values.
The final step consists of linearly projecting back the *TokenStats* (referred to as *ProjStats*) to 2 scalar values that are then used to denormalize the output, providing the final individual appliance consumption.

#### TimeRPE

The Transformer architecture does not inherently understand sequence order due to its self-attention mechanisms, which are permutation invariant. 
Therefore, Positional Encoding (PE) is mandatory to provide this context, allowing the model to consider the position of each token in a sequence. 
Fixed sinusoidal or fully learnable PEs are commonly used in most current Transformer-based architectures for time series analysis (PatchTST), including those proposed for energy disaggregation (BERT4NILM, Energformer, STNILM). 
This kind of PE consists of adding a matrix of fixed or learnable weight on the extracted features before the Transformer block.
However, these PEs only help the model understand local context information (i.e., the given order of the tokens in the sequence) and do not provide any information about the global context when operating on subsequences of a longer series. 
In the context of NILM, appliance use is often related to specific periods (e.g., dishwashers running after mealtimes, electric vehicles charging at night, or on weekends). 
Moreover, detailed timestamp information is always available in real-world NILM applications.
Thus, using a PE based on timestamp information can help the model better understand the recurrent use of appliances. 
Timestamp-based PEs have been briefly investigated for time series forecasting but were always combined with a fixed or learnable PE and directly added to the extracted features.

Therefore, we proposed the Timestamps Related Positional Encoding (TimeRPE), a Positional Encoding based only on the discrete timestamp values extracted from the input subsequences.
The TimeRPE module, depicted in Figure~\ref{fig:nilmformerparts} (c), takes as input the timestamps information $t$ from the input subsequences, decomposes it such as minutes $t^m$, hours $t^h$, days $t^d$, and months $t^M$, and project them in a sinusoidal basis, as:

```math
    T_{\sin}(t_i) = \sin \left(\frac{2 \pi t^j_i}{p^j} \right) \quad \text{and} \quad
    T_{\cos}(t_i) = \cos \left(\frac{2 \pi t^j_i}{p^j} \right)$,
```
with $j \in \{m, h, d, M\}$ and $\{p^m=59, p^h=23, p^d=6, p^M=11\}$ corresponding to the set of max possible discrete timestamp variable.
Afterward, the obtained representation is projected in a higher dimensional space using a 1D convolution layer with a kernel of size 1.


## Contributors

* Adrien Petralia (Université Paris Cité, EDF Research)
* Philippe Charpentier (EDF Research)
* Youssef Kadhi Charpentier (EDF Research)
* Themis Palpanas (IUF, Université Paris Cité) 

