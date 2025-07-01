# Dynamic programming with Tensorflow and Keras

This Python-based code repository supplements the work of [Galo Nuno](https://www.galonuno.com/), [Philipp Renner](https://www.lancaster.ac.uk/lums/people/philipp-renner) and [Simon Scheidegger](https://sites.google.com/site/simonscheidegger), titled _[Monetary Policy with Persistent Supply Shocks](#citation)_ (Nuno, Renner and Scheidegger; 2025), which introduces a highly scalable computational technique to solve macro prudential policy models.

However, the scope of the method is much broader: as it is a generic framework to compute global solutions to dynamic (stochastic) models with many state variables, it is applicable to almost any dynamic model. The available solution algorithms are based on "policy function iteration".


* The computational framework located [here](DEQN) is extensively documented, leverages [Tensorflow](https://docs.gpytorch.ai/en/v1.5.1/index.html), and combines Artifical Neural Network regression with performance-boosting options such as GPU parallelism.
* Replication codes for the macro prudential models from the paper are provided.


### Authors
* [Galo Nuno](https://www.galonuno.com/) (Bank of Spain)
* [Philipp Renner](https://www.lancaster.ac.uk/lums/people/philipp-renner) (University of Lancaster, Department of Economics)
* [Simon Scheidegger](https://sites.google.com/site/simonscheidegger) (University of Lausanne, Department of Economics)

### Citation

Please cite [ Nuño, Galo and Renner, Philipp Johannes and Scheidegger, Simon, Monetary Policy with Persistent Supply Shocks (October 30, 2024).](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5005047)
in your publications if it helps your research:

## Usage
We provide implementations which use python 3.
