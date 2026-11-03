# Literature index

The papers the brief cites as `[paper: <name>]`. Originals are in `sources/` and plain-text extractions (pdftotext, for
search) in `text/`; both folders are gitignored, so this index is the committed record. Formulas are unreliable in the
text extractions: check them in the original.

To add a source: put the original in `sources/<key>.<ext>`, extract with `pdftotext sources/<key>.pdf text/<key>.txt`,
and add a row here with its sha256. A source that is no longer cited in the brief is removed from all three places.

| Key | Brief name | Reference | Cited in brief | sha256 |
|---|---|---|---|---|
| bouthillier2021 | Bouthillier | Bouthillier et al. 2021, Accounting for Variance in Machine Learning Benchmarks, MLSys. arXiv:2103.03098 | S7, S8, S12, C7 | 1f458f7e606e4db86b207315c049a38b52cacde15ad34def2cc6c29124fc5d4c |
| chizat2019 | Chizat | Chizat, Oyallon and Bach 2019, On Lazy Training in Differentiable Programming, NeurIPS. arXiv:1812.07956 | C2, G7, G13 | 70299b331e75c305c2d54597c82c135ebc1f55d2c7c3f03793abc2588aa31387 |
| choi2019 | Choi | Choi et al. 2019, On Empirical Comparisons of Optimizers for Deep Learning. arXiv:1910.05446 | S7, C2, G9, G10 | da07a208f2843eaaa2363078de41bf8a2a4dee7e2067f21b0863b4787f69c412 |
| defazio2024 | Defazio | Defazio et al. 2024, The Road Less Scheduled, NeurIPS. arXiv:2405.15682 | S5, S7, S9, G8, G9, G10, G11 | 1aef42351ac417eb08ba81919cb49bd9d14d98320d0bd3d47eac72f48dcf20d3 |
| gu2020 | Gu, Kelly and Xiu | Gu, Kelly and Xiu 2020, Empirical Asset Pricing via Machine Learning, Review of Financial Studies 33(5). doi:10.1093/rfs/hhaa009 | G2, G4 | bc637ae801a45778ad127291abba7e49b1ff4e772753fbd832f145c17a8daeb7 |
| guo2008 | Guo | Guo 2008, Regionalization with Dynamically Constrained Agglomerative Clustering and Partitioning (REDCAP), IJGIS 22(7). doi:10.1080/13658810701674970 | S12, G15 | ea9c5c6cbdeb0237c6523096be10cff75c8a1d8eba2403dcf50adad960bd5750 |
| hutter2014 | fANOVA | Hutter, Hoos and Leyton-Brown 2014, An Efficient Approach for Assessing Hyperparameter Importance, ICML | S11 | 496cd495c27388943af3747b6a4b35c42fbe79a046c9746d715f5bb36fbbd5b9 |
| kunin2024 | Kunin | Kunin et al. 2024, Get Rich Quick: Exact Solutions Reveal How Unbalanced Initializations Promote Rapid Feature Learning, NeurIPS | C2, G7 | 2a1774f20b4af5c11e4f01117c5f0746c87770e9c1ce5701a019df5400c94e63 |
| lakens2017 | Lakens | Lakens 2017, Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses, SPPS 8(4). doi:10.1177/1948550617697177 | S8, S12, G11, G15 | e562637d095c290e88d9981691fba03d372b53cc45261df37736f9838103d168 |
| li2024 | Li et al. | Li et al. 2024, Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling, NeurIPS | S7, S11, S13, G9 | 27e04fa74ef7901e76fd7facc5bf2cd373566072aa8b664805ecce3aaddd9bac |
| liu2022 | Liu | Liu et al. 2022, Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting, NeurIPS. arXiv:2205.14415 | S2, C2, G1 | 60797bc736ba73f1c6aef5200dd97b9e6d9fdec64b0ef2cfd0ee848a0a5db2a7 |
| neal2018 | Neal | Neal et al. 2018, A Modern Take on the Bias-Variance Tradeoff in Neural Networks. arXiv:1810.08591 | S9, G4, G12 | 7c37bcc8f935e4d2f503b087c94ce7c4879daa1f49bb31fc45d1dbe017559830 |
| playbook | Playbook | Godbole et al. 2023, Deep Learning Tuning Playbook, github.com/google-research/tuning_playbook | S7, C6, G5, G6, G9 | 9ee275082cce3b7b1b49d1bfc7b1d6c0e2b4e6a415b6c67a73a72e2eedcd6187 |
| radosavovic2020 | RegNet | Radosavovic et al. 2020, Designing Network Design Spaces, CVPR. arXiv:2003.13678 | S12 | ea2d65be47f5a29aaf0de0893a57aff5cbb0c4c8ada6cda89adc9da4d8f1c4d3 |
| rhif2019 | Rhif | Rhif et al. 2019, Wavelet Transform Application for/in Non-Stationary Time-Series Analysis: A Review, Applied Sciences 9(7). doi:10.3390/app9071345 | G1 | d31d52bf3c2bfffc76cdccb7a6afaa5242227bd8fb3073d1acc4673d25360892 |
| shallue2018 | Shallue | Shallue et al. 2019, Measuring the Effects of Data Parallelism on Neural Network Training, JMLR 20. arXiv:1811.03600 | C6, G6 | 6833f0a479f2727920889e479516e90e2bdc40032525385dd1ef133ad916bdfc |
| summers2021 | Summers | Summers and Dinneen 2021, Nondeterminism and Instability in Neural Network Optimization, ICML (PMLR 139) | S7, S8, G4, G12 | 8f7fb2f0c2e0409241573eaf2c5cfede9965e43c04f35c30ff0bfd0c43beeded |
| tawakuli2024 | Tawakuli | Tawakuli et al. 2024, Survey: Time-Series Data Preprocessing, Journal of Engineering Research 13. doi:10.1016/j.jer.2024.02.018 | kept as a guide, not cited | b714ea13b6fc4992d62858efb4846db988de7c7901f82548ee84531c4edcf036 |
| xu2021 | Xu | Xu et al. 2021, How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks, ICLR | C2 | e8addcbd3bb71f320b009af7778e97331184a4dfd0349acc71254623423f1f86 |
