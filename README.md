<div align="center">
  <img src="assets/logo.png" alt="LAMINR Logo" width="100%">
</div>
<br>

**LAMINR** (**L**earning and **A**ligning **M**anifolds of Single-Neuron Invariances using **I**mplicit **N**eural **R**epresentations) enables the **systematic discovery and alignment of invariance manifolds** in stimulus space for visual sensory neurons, providing a principled way to characterize and compare neuronal invariances at the **population level**, independent of nuisance receptive field properties such as position, size, and orientation.

### 🚀 Highlights

- **Continuous Invariance Manifold Learning:** Identifies the full space of stimuli that elicit near-maximal responses from a neuron.
- **Alignment Across Neurons:** Learns transformations that align invariance manifolds across neurons, revealing shared invariance properties.
- **Functional Clustering:** Uncovers distinct functional neuron clusters based on their invariance properties.
- **Model-Agnostic:** Can be applied to any robust response-predicting model of biological neurons.

<div align="center">
  <img src="assets/method.png" alt="Method Overview" width="90%">
</div>

## 🛠 Installation

You can install LAMINR using one of the following methods:

### 1️⃣ Using `pip`
```bash
pip install laminr
```

### 2️⃣ Via GitHub (Latest Version)
```bash
pip install git+https://github.com/your-org/laminr.git
```

## 🔥 Quick Start

Here's a simple example of how to use **LAMINR** to learn and align invariance manifolds.

```python
import laminr

# Load a trained response-predicting model
model = laminr.load_model("macaque_v1")

# Select neurons of interest
neuron1, neuron2 = model.get_neurons(42, 73)

# Learn the invariance manifold for neuron1
manifold1 = laminr.learn_invariance_manifold(neuron1)

# Align with neuron2 to quantify shared invariance properties
alignment_score, transformed_manifold = laminr.align_manifolds(manifold1, neuron2)

print(f"Alignment Score: {alignment_score:.3f}")
```

For a more detailed walkthrough, see our [examples](examples).

## 🛠 Questions & Contributions

If you encounter any issues while using the method, please create an [Issue](https://github.com/your-org/laminr/issues) on GitHub.

We welcome and appreciate contributions to the package! Feel free to open an [issue](https://github.com/your-org/laminr/issues) or submit a [pull request](https://github.com/your-org/laminr/pulls) for new features.

For other questions or project collaboration inquiries, please contact mohammadbashiri93@gmail.com or loocabaroni@gmail.com.

## 📜 License

This package is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License**. Briefly:
- **Attribution Required**: You must credit the original authors and indicate if changes were made.
- **NonCommercial Use Only**: This package may not be used for commercial purposes without explicit permission.
- **No Additional Restrictions**: You may not apply legal terms that prevent others from using this package under these terms.

For full details, see the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).<br>
For commercial use inquiries, please contact: mohammadbashiri93@gmail.com.

## 📖 Paper

**ICLR 2025** (\<PRESENTATION FORMAT\>): ["Learning and Aligning Single-Neuron Invariance Manifolds in Visual Cortex"](https://openreview.net/forum?id=kbjJ9ZOakb) <br>
**Authors**: Mohammad Bashiri*, Luca Baroni*, Ján Antolík, Fabian H. Sinz. (* denotes equal contribution)

Please cite our work if you find it useful:

```bibtex
@inproceedings{bashiri2025laminr,
  title={Learning and Aligning Single-Neuron Invariance Manifolds in Visual Cortex},
  author={Bashiri, Mohammad and Baroni, Luca and Antolík, Ján and Sinz, Fabian H.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## 📝 To-Do

- [ ] Add full documentation
- [ ] Provide pretrained models
- [ ] Add a link to google colab for running the quick start example
- [ ] Add some light-weight gifs (optional - we need to make sure the git repo is as light-weight as possible)
- [ ] Include a web-based visualization tool for invariance manifolds (optional)
