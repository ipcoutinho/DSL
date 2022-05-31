# DSL - Transformer-based Models for ICD Coding of Clinical Text

Transformer-based Language Models have become the go-to approach for most NLP tasks. However, in the medical domain, their performance still lags behind tailored approaches based on CNNs or RNNs, particularly in the ICD coding task. We hypothesise that there are two reasons for this: text truncation and domain mismatch. While standard Transformers are unable to process long texts and lack exposure to domain-specific data, clinical notes are typically lengthy and contain a vast medical vocabulary. In this work, we empirically demonstrate that pretraining a Transformer capable of processing long documents using in-domain data achieves promising results on a subset of the MIMIC-III dataset. Additionally, we explore the use of different loss functions in order to mitigate the effect of the label distribution being imbalanced, showing that the use of an asymmetric loss can improve performance on most metrics.

The experiments were conducted on the MIMIC-III dataset (Johnson et al., 2016), specifically using the subset with the most frequent 50 codes, following the splits of dataset made publicly available by Mullenbach et al. (2018) at <a href="http://github.com/jamesmullenbach/caml-mimic">github.com/jamesmullenbach/caml-mimic</a>.
To measure the predictive capability of the model, we use the evaluation.py script also made aavailable by Mullenbach et al. (2018).

<pre><code>
@article{Johnson2016,
    author = {Johnson, Alistair E.W. and Pollard, Tom J. and Shen, Lu and Lehman, Li Wei H. and Feng, Mengling and Ghassemi, Mohammad and Moody, Benjamin and Szolovits, Peter and Anthony Celi, Leo and Mark, Roger G.},
    journal = {Scientific Data},
    pages = {1--9},
    title = {MIMIC-III, a freely accessible critical care database},
    volume = {3},
    year = {2016}
}
</code></pre>
