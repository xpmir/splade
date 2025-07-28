# SPLADE

This is the experiment files for SPLADE models, developed within [experimaestro-ir (XPMIR)](https://github.com/experimaestro/experimaestro-ir).
The papers are described in [Towards Effective and Efficient Sparse Neural Information Retrieval](https://dl.acm.org/doi/10.1145/3634912):

<details>
    <summary><i>BibTex record</i></summary>

    ```bibtex
    @article{10.1145/3634912,
    author = {Formal, Thibault and Lassance, Carlos and Piwowarski, Benjamin and Clinchant, St\'{e}phane},
    title = {Towards Effective and Efficient Sparse Neural Information Retrieval},
    year = {2023},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    issn = {1046-8188},
    url = {https://doi.org/10.1145/3634912},
    doi = {10.1145/3634912},
    abstract = {Sparse representation learning based on Pre-trained Language Models has seen a growing interest in Information Retrieval. Such approaches can take advantage of the proven efficiency of inverted indexes, and inherit desirable IR priors such as explicit lexical matching or some degree of interpretability. In this work, we thoroughly develop the framework of sparse representation learning in IR, which unifies term weighting and expansion in a supervised setting. We then build on SPLADE – a sparse expansion-based retriever – and show to which extent it is able to benefit from the same training improvements as dense bi-encoders, by studying the effect of distillation, hard negative mining as well as the Pre-trained Language Model’s initialization on its effectiveness – leading to state-of-the-art results in both in- and out-of-domain evaluation settings (SPLADE++). We furthermore propose efficiency improvements, allowing us to reach latency requirements on par with traditional keyword-based approaches (Efficient-SPLADE).},
    note = {Just Accepted},
    journal = {ACM Trans. Inf. Syst.},
    month = {dec},
    keywords = {Sparse Representations, Information Retrieval, Efficiency, Effectiveness}
    }
    ```
</details>


## Run it


The repository is build based on the [experimaestro-ir](https://github.com/experimaestro/experimaestro-ir) framework, which is a repository providing many basic key components in doing IR experiments, including learning, indexing, evaluating, etc. Our dataset access is done under the [ir-datasets](https://github.com/allenai/ir_datasets/) package, through the interface of [datamaestro](https://github.com/experimaestro/datamaestro_text).

We use [`uv`](https://docs.astral.sh/uv/) for dependency management. Once installed, you can just use `uv run` within the cloned repository and this will install all the necessary depenencies.

**Important**: When setting the experimaestro experiments, you also need to prepare a `launcher.py` under the folder `~/.config/experimaestro`. Examples could be found [here](https://experimaestro-python.readthedocs.io/en/latest/launchers/).


```sh
# See below for the list of possible
uv run experimaestro run-experiment splade/NAME.yaml
```

The implemented models are:
- `splade/normal_DistilMSE`: SPLADE-max with distillation (one of the best performing model)
- `splade/debug`: SPLADE-max with distillation (one of the best performing model), just performing a few learning step and indexing a part of the documents