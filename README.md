# Transformer-based Models for ICD Coding of Clinical Text

Transformer-based Language Models have become the go-to approach for most NLP tasks. Nonetheless, in the medical domain, their performance still lags behind tailored approaches based on CNNs or RNNs, particularly in the ICD coding task. We hypothesise that there are mainly two reasons for this: text truncation and domain mismatch. Standard Transformers are input length limited and trained on general purpose data, making them suitable for small text tasks without special vocabulary. However, clinical notes are typically lengthy and contain a vast medical vocabulary, making these models unable to process these texts successfully. In this work, we empirically demonstrate that pretraining a Transformer capable of processing long documents using in-domain data achieves promising results on a subset of the MIMIC-III dataset. Additionally, we explore the use of different loss functions in order to mitigate the effect of the label distribution being imbalanced, showing that the use of an asymmetric loss can improve performance on most metrics.

The experiments were conducted on the MIMIC-III dataset (Johnson et al., 2016), specifically using the subset with the most frequent 50 codes, following the splits of dataset made publicly available by Mullenbach et al. (2018) at <a href="http://github.com/jamesmullenbach/caml-mimic">github.com/jamesmullenbach/caml-mimic</a>. 

The implementation of the asymmetric loss (Ben-Baruch et al., 2020) described  in the paper is available at  <a href="https://github.com/Alibaba-MIIL/ASL">https://github.com/Alibaba-MIIL/ASL</a>.

To measure the predictive capability of the model, we use the evaluation.py script also made available by Mullenbach et al. (2018).

<pre><code>
@article{Johnson2016,
    author = {Johnson, Alistair E.W. and Pollard, Tom J. and Shen, Lu and Lehman, Li Wei H. and Feng, Mengling and Ghassemi, Mohammad and Moody, Benjamin and Szolovits, Peter and Anthony Celi, Leo and Mark, Roger G.},
    journal = {Scientific Data},
    pages = {1--9},
    title = {MIMIC-III, a freely accessible critical care database},
    volume = {3},
    year = {2016}
    
@article{Ben2020,
    title={Asymmetric loss for multi-label classification},
    author={Ben-Baruch, Emanuel and Ridnik, Tal and Zamir, Nadav and Noy, Asaf and Friedman, Itamar and Protter, Matan and Zelnik-Manor, Lihi},
    journal={arXiv preprint arXiv:2009.14119},
    year={2020}
}

@inproceedings{Mullenbach2018,
    title={Explainable Prediction of Medical Codes from Clinical Text},
    author={Mullenbach, James and Wiegreffe, Sarah and Duke, Jon and Sun, Jimeng and Eisenstein, Jacob},
    booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
    pages={1101--1111},
    year={2018}
}
</code></pre>
