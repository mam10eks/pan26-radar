# PAN-Shared-Tasks-Generative-AI-Detection-2026

# Voight-Kampff Generative AI Detection 2026

* [Synopsis](https://pan.webis.de/clef26/pan26-web/generated-content-analysis.html#synopsis)  
* [Task Overview](https://pan.webis.de/clef26/pan26-web/generated-content-analysis.html#task-overview)  
* [Data](https://pan.webis.de/clef26/pan26-web/generated-content-analysis.html#data)  
* [Submission](https://pan.webis.de/clef26/pan26-web/generated-content-analysis.html#submission)  
* [Evaluation](https://pan.webis.de/clef26/pan26-web/generated-content-analysis.html#evaluation)  
* [Baselines](https://pan.webis.de/clef26/pan26-web/generated-content-analysis.html#baselines)  
* [Leaderboard](https://pan.webis.de/clef26/pan26-web/generated-content-analysis.html#leaderboard)  
* [Related Work](https://pan.webis.de/clef26/pan26-web/generated-content-analysis.html#related-work)  
* 

## Synopsis

* Task: Given a (potentially obfuscated) text, decide whether it was written by a human or an AI.  
* Registration: \[[CLEF labs](https://clef-labs-registration.dipintra.it/)\] \[[Tira](https://www.tira.io/task-overview/generative-ai-authorship-verification-panclef-2026/)\]  
* Important dates:  
  * **May 07, 2026 May 21, 2026:** software submission  
  * **May 28, 2026:** participant notebook submission \[[template](https://pan.webis.de/pan-notebook-paper-template/pan-notebook-paper-template.zip)\] \[[submission](https://easychair.org/conferences/?conf=clef2026)  – *select "Stylometry and Digital Text Forensics (PAN)"* \]  
* Data: Human and machine texts \[[download](https://zenodo.org/records/14962653)\]  
* Evaluation Measures: F1, C@1, AUC-ROC, FPR, FNR  
* Baselines: SVM, Compression, Binoculars \[[code](https://github.com/pan-webis-de/pan-code/tree/master/clef25/generative-authorship-verification/pan25_genai_baselines)\]

## Task Overview

The Voight-Kampff AI Detection task is a binary AI detection task in that participants are given a text and have to decide whether it was machine-authored (class 1\) or human-authored (class 0). However, we introduced a twist: The LLMs were instructed to change their style and mimic a specific human author. Furthermore, the test set will contain several surprises such as new models or unknown obfuscations to test the robustness of the classifiers (however, texts will be from the same domain).

As in the previous year, the *Voight-Kampff AI detection Task* @ PAN is organized in collaboration with the *Voight-Kampff Task* @ [ELOQUENT Lab](https://eloquent-lab.github.io/) Lab in a builder-breaker style. PAN participants will build systems to tell human and machine apart, while ELOQUENT participants will investigate novel text generation and obfuscation methods for avoiding detection.

## Data

The dataset is available via [Zenodo](https://zenodo.org/records/14962653). The dataset contains copyrighted material and may be used only for research purposes. **No redistribution allowed.**

The training and validation dataset is provided as a set of newline-delimited JSON files. Each file contains a list of texts, written either by a human or a machine. The file format is as follows:

{"id": "a6c8018e-d22c-4d6e-b5e3-0c0a65682a6a", "text": "...", "model": "human", "label": 0, "genre": "essays"}  
{"id": "f1a26761-ca2a-43e9-890d-80dcb3058364", "text": "...", "model": "gpt-4o", "label": 1, "genre": "essays"}

...

A "label" of 0 means human-written, 1 is ai-written. "genre" is for informational purposes only and can be either "essays", "news", or "fiction". Texts with "genre": "news" are sampled from last year's dataset (but with a few additions, such as GPT-4o). So if you want to reuse last year's dataset, be aware that some texts will be duplicates\!

The test dataset will have the same format, but with only the "id" and "text" columns.

## Submission

Participants will submit their systems as Docker images through the [Tira](https://www.tira.io/task-overview/generative-ai-authorship-verification-panclef-2026/) platform. It is not expected that submitted systems are actually *trained* on Tira, but they must be standalone and runnable on the platform without requiring contact to the outside world (evaluation runs will be sandboxed).

The submitted software must be executable inside the container via a command line call. The script must take two arguments: an input file (an absolute path to the input JSONL file) and an output directory (an absolute path to where the results will be written):

Within Tira, the input file will be called dataset.jsonl, so with the pre-defined Tira placeholders, your software should be invoked like this:

$ mySoftware $inputDataset/dataset.jsonl $outputDir

Within $outputDir, a single (\!) file with the file extension \*.jsonl must be created with the following format:

{"id": "bea8cccd-0c99-4977-9c1b-8423a9e1ed96", "label": 1.0}  
{"id": "a963d7a0-d7e9-47c0-be84-a40ccc2005c7", "label": 0.2315}

...

For each test case in the input file, an output line must be written with the ID of the input text pair and a confidence score between 0.0 and 1.0. A score \< 0.5 means that the text is believed to be human-authored. A score \> 0.5 means that it is likely machine-written. A score of *exactly* 0.5 means the case is undecidable. Participants are encouraged to answer with 0.5 rather than making a *wrong* prediction. You can also give binary score (0 and 1\) if your detector does not output class probabilities.

**All test cases must be processed in isolation without information leakage between them\!** Even though systems may be given an input file with multiple JSON lines at once for reasons of efficiency, these inputs must be processed and answered just the same as if only a single line were given. Answers for any one test case must not depend on other cases in the input dataset\!

## Evaluation

Systems will be evaluated with the same measures as previous installments of the PAN authorship verification tasks. The following metrics will be used:

* ROC-AUC: The area under the ROC (Receiver Operating Characteristic) curve.  
* Brier: The complement of the Brier score (mean squared loss).  
* C@1: A modified accuracy score that assigns non-answers (score \= 0.5) the average accuracy of the remaining cases.  
* F1: The harmonic mean of precision and recall.  
* F0.5u: A modified F0.5 measure (precision-weighted F measure) that treats non-answers (score \= 0.5) as false negatives.  
* The arithmetic mean of all the metrics above.  
* A confusion matrix for calculating true/false positive/negative rates.

The evaluator for the task will output the above measures as JSON like so:

{  
    "roc-auc": 0.996,  
    "brier": 0.951,  
    "c@1": 0.984,  
    "f1": 0.98,  
    "f05u": 0.981,  
    "mean": 0.978,  
    "confusion": \[  
        \[  
            1211,  
            66  
        \],  
        \[  
            27,  
            2285  
        \]  
    \]

}

## Baselines

We provide three LLM detection baselines:

* Linear SVM with TF-IDF features  
  *Validation:* \[ROC-AUC: 0.996; Brier: 0.951; C@1: 0.984; F1: 0.980; F0.5u: 0.981; Mean: 0.978\]  
* PPMd Compression-based Cosine \[[Sculley and Brodley, 2006](https://ieeexplore.ieee.org/abstract/document/1607268)\] \[[Halvani et al., 2017](https://dl.acm.org/doi/abs/10.1145/3098954.3104050)\]  
  *Validation:* \[ROC-AUC: 0.786; Brier: 0.799; C@1: 0.757; F1: 0.812; F0.5u: 0.778; Mean: 0.786\]  
* Binoculars \[[Hans et al., 2024](https://arxiv.org/abs/2401.12070)\]  
  *Validation:* \[ROC-AUC: 0.918; Brier: 0.867; C@1: 0.844; F1: 0.872; F0.5u: 0.882; Mean: 0.877\]

With TF-IDF SVM and PPMd CBC, we provide two bag-of-words authorship verification models. Binoculars uses large language models to measure text perplexity. The SVM classifier is a supervised LLM detector, the other two are unsupervised / zero-shot models. The baselines are published on [GitHub](https://github.com/pan-webis-de/pan-code/tree/master/clef25/generative-authorship-verification/pan25_genai_baselines). You can run them locally, in a Docker container, or using tira-run. All baselines come with a CLI and usage instructions. Their general usage is:

$ pan25-baseline BASELINENAME INPUT\_FILE OUTPUT\_DIRECTORY

Use \--help on any subcommand for more information:  
$ pan25-baseline \--help  
Usage: pan25-baseline \[OPTIONS\] COMMAND \[ARGS\]...

  PAN'25 Generative AI Authorship Verification baselines.

Options:  
  \--help  Show this message and exit.

Commands:  
  binoculars  PAN'25 baseline: Binoculars.  
  ppmd        PAN'25 baseline: Compression-based cosine.

  tfidf       PAN'25 baseline: TF-IDF SVM.

More information on how to install and run the baselines can be found in the [README on GitHub](https://github.com/pan-webis-de/pan-code/tree/master/clef25/generative-authorship-verification/pan25_genai_baselines).

## Leaderboard

TBD

## Related Work

* Janek Bevendorff et al. [Overview of the “Voight-Kampff” Generative AI Authorship Verification Task at PAN and ELOQUENT 2025\.](https://webis.de/publications.html#bevendorff_2025d) In Guglielmo Faggioli, Nicola Ferro, Paolo Rosso, and Damiano Spina, editors, Working Notes of CLEF 2025 – Conference and Labs of the Evaluation Forum, CEUR Workshop Proceedings, pages 3504–3534, September 2025\. CEUR-WS.org.  
* Janek Bevendorff, Matti Wiegmann, Jussi Karlgren, Luise Dürlich, Evangelia Gogoulou, Aarne Talman, Efstathios Stamatatos, Martin Potthast, and Benno Stein. [Overview of the “Voight-Kampff” Generative AI Authorship Verification Task at PAN and ELOQUENT 2024\.](https://webis.de/publications.html#bevendorff_2024d) In Guglielmo Faggioli, Nicola Ferro, Petra Galuščáková, and Alba García Seco de Herrera, editors, Working Notes of CLEF 2024 – Conference and Labs of the Evaluation Forum, CEUR Workshop Proceedings, pages 2486-2506, September 2024\. CEUR-WS.org.  
* Janek Bevendorff, Matti Wiegmann, Emmelie Richter, Martin Potthast, and Benno Stein. [The Two Paradigms of LLM Detection: Authorship Attribution vs. Authorship Verification.](https://webis.de/publications.html#bevendorff_2025b) Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, The 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025\) (Findings), pages 3762–3787, July 2025\. Association for Computational Linguistics.  
* Bevendorff, Janek, Xavier Bonet Casals, Berta Chulvi, Daryna Dementieva, Ashaf Elnagar, Dayne Freitag, Maik Fröbe, et al. 2024\. [Overview of PAN 2024: Multi-Author Writing Style Analysis, Multilingual Text Detoxification, Oppositional Thinking Analysis, and Generative AI Authorship Verification: Extended Abstract.](https://webis.de/publications.html#bevendorff_2024b) In Lecture Notes in Computer Science, 3-10. Lecture Notes in Computer Science. Cham: Springer Nature Switzerland.  
* Uchendu, Adaku, Thai Le, Kai Shu, and Dongwon Lee. 2020\. [Authorship Attribution for Neural Text Generation.](https://aclanthology.org/2020.emnlp-main.673/) In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 8384-95. Online: Association for Computational Linguistics.  
* Jakesch, Maurice, Jeffrey T. Hancock, and Mor Naaman. 2023\. [Human Heuristics for AI-Generated Language Are Flawed.](https://www.pnas.org/doi/10.1073/pnas.2208839120) Proceedings of the National Academy of Sciences of the United States of America 120 (11): e2208839120.  
* Hans, Abhimanyu, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024\. [Spotting LLMs with Binoculars: Zero-Shot Detection of Machine-Generated Text.](http://arxiv.org/abs/2401.12070) arXiv \[Cs.CL\].  
* Su, Jinyan, Terry Yue Zhuo, Di Wang, and Preslav Nakov. 2023\. [DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text.](https://arxiv.org/abs/2306.05540) arXiv \[Cs.CL\].  
* Mitchell, Eric, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, and Chelsea Finn. 2023\. [DetectGPT: Zero-Shot Machine-Generated Text Detection Using Probability Curvature.](http://arxiv.org/abs/2301.11305) arXiv \[Cs.CL\].  
* Bao, Guangsheng, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. 2023\. [Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature.](https://arxiv.org/abs/2310.05130) arXiv \[Cs.CL\].  
* Koppel, Moshe, and Jonathan Schler. 2004\. [Authorship Verification as a One-Class Classification Problem.](https://dl.acm.org/doi/abs/10.1145/1015330.1015448) In Proceedings, Twenty-First International Conference on Machine Learning, ICML 2004, 489-95.  
* Bevendorff, Janek, Benno Stein, Matthias Hagen, and Martin Potthast. 2019\. [Generalizing Unmasking for Short Texts.](https://webis.de/publications.html#bevendorff_2019a) In Proceedings of the 2019 Conference of the North, 654-59. Stroudsburg, PA, USA: Association for Computational Linguistics.  
* Sculley, D., and C. E. Brodley. 2006\. [Compression and Machine Learning: A New Perspective on Feature Space Vectors.](https://ieeexplore.ieee.org/abstract/document/1607268) In Data Compression Conference (DCC'06), 332-41. IEEE.  
* Halvani, Oren, Christian Winter, and Lukas Graner. 2017\. [On the Usefulness of Compression Models for Authorship Verification.](https://dl.acm.org/doi/abs/10.1145/3098954.3104050) In ACM International Conference Proceeding Series. Vol. Part F1305. Association for Computing Machinery. https://doi.org/10.1145/3098954.3104050.  
* Uchendu, Adaku, Zeyu Ma, Thai Le, Rui Zhang, and Dongwon Lee. 2021\. [TURINGBENCH: A Benchmark Environment for Turing Test in the Age of Neural Text Generation.](https://aclanthology.org/2021.findings-emnlp.172/) In Findings of the Association for Computational Linguistics: EMNLP 2021, 2001-16. Stroudsburg, PA, USA: Association for Computational Linguistics.  
* Schuster, Tal, Roei Schuster, Darsh J. Shah, and Regina Barzilay. 2020\. [The Limitations of Stylometry for Detecting Machine-Generated Fake News.](https://direct.mit.edu/coli/article/46/2/499/93369/The-Limitations-of-Stylometry-for-Detecting) Computational Linguistics 46 (2): 499-510.  
* Sadasivan, Vinu Sankar, Aounon Kumar, Sriram Balasubramanian, Wenxiao Wang, and Soheil Feizi. 2023\. [Can AI-Generated Text Be Reliably Detected?](http://arxiv.org/abs/2303.11156) arXiv \[Cs.CL\].  
* Ippolito, Daphne, Daniel Duckworth, Chris Callison-Burch, and Douglas Eck. 2020\. [Automatic Detection of Generated Text Is Easiest When Humans Are Fooled.](https://aclanthology.org/2020.acl-main.164/) In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 1808-22. Stroudsburg, PA, USA: Association for Computational Linguistics.

