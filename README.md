# Automatic Romanian Text Generation using GPT-2

One of the most significant tasks in natural language processing (NLP) is text generation, which beneficiate from the recent architectures that use large pre-trained transformer models, such as the Generative Pre-trained Transformer-2 ([GPT-2](https://github.com/openai/gpt-2)) or GPT-3 developed by OpenAI, and Google’s Bidirectional Encoder Representations from Transformers ([BERT](https://aclanthology.org/N19-1423/)). The paper presents a NLG model based on the GPT-2 architecture that generates Romanian online news, using manually annotated texts. A small Romanian GPT-2 model, using 24 thousand news items, named MCBGPT-2 was developed, tested and evaluated. Additionally, an existing Romanian GPT-2 model, called [RoGPT-2](https://huggingface.co/readerbench/RoGPT2-base), was added to experiments. For evaluation, is presented a comparison of several automatic metrics such as [BLEU](https://dl.acm.org/doi/10.3115/1073083.1073135), [ROUGE](https://aclanthology.org/W04-1013), [BLEURT](https://aclanthology.org/2020.acl-main.704/) and [BERTScore](https://arxiv.org/abs/1904.09675) applied to generated news items from the test and validation datasets. Experimental results revealed that the MCBGPT-2 and RoGPT-2 models provided similar performances in text generation task for Romanian language, using less data for MCBGPT-2 model’s training process. 

**Dataset description**

Figure 1 presents the top news sources and the distribution of number of articles from each source, such as agerpres.ro – 2320, monitorulapararii.ro – 720, news.ro – 614 or evz.ro – 536 articles.
The aforementioned dataset was split into train (60%), validation (20%), and test (20%).  The training process of the MCBGPT-2 model used 14,756 news items, while 4922 news items in each of the test and validation datasets were presented. 

Table 1 and Figure 2 presents the distributions of the test, validation, and train datasets, revealing that our datasets contain the same average words per news item (e.g., ~200 words) and the vocabulary size (unique words from The Explanatory Dictionary of the Romanian Language - DEX) of the test and validation datasets are well balanced (e.g., ~1M words), while the training dataset is about three times larger (e.g., ~3M words).

<img alt="Figure 1" title="Figure 1 - Distribution of articles across news sources" src="https://user-images.githubusercontent.com/114503442/192862531-be93a66a-c36e-4479-aa99-c543d06d9a92.png" width="200" height="200"> <img alt="Figure 2" title="Figure 2 - Distribution of words across test, validation and training dataset" src="https://user-images.githubusercontent.com/114503442/192862554-41753d5c-d8d2-4f22-9378-3a2ae9a466f4.png" width="200" height="200">

**Table 1 Distribution of words across test, validation and train datasets**
|        Words                  |   Dataset   |     No     |        Words                  |   Dataset   |     No     | 
| ----------------------------- |:-----------:|:----------:| ----------------------------- |:-----------:|:----------:| 
|                               |    Test     | 957,787    |                               |    Test     |    194     |
|    Romanian unique words      | Validation  | 1,026,680  |  Average words  per news item | Validation  |    208     |
|                               |   Train     | 2,837,236  |                               |    Train    |    192     |

The experiments were performed on a single server using tf.distribute.Mirrored Strategy, a TensorFlow API to distribute training across multiple GPUs and several software or packages such as CUDA (vers. 11.2), Python (vers. 3.9.7) and TensorFlow (vers. 2.4.1).The proposed MCBGPT-2 model was trained and validated using the aforementioned news corpus, that contain 18 fake news and 24,582 true news, 368 news with negative polarity and 2135 news with positive polarity, being split in equal parts between train, test and validation datasets. For this research, the Adam optimizer is used with a small learning rate (e.g., 3 x 105), a loss function such as sparse categorical cross-entropy and a vocabulary size of 60,000 words. Several training statistics of the proposed model, including the hyperparameters and training time, are presented in Table 2. 

**Table 2 Training statistics of MCBGPT-2 model**
|    Parameters name   | Value of parameter | 
| -------------------- |:------------------:| 
| Number of parameters |       131M         | 
|    Number of epoch   |        15          |  
| Duration of an epoch |        5h          | 
|     Context size     |       512          | 
|      Batch size      |        12          | 

**Experiments**

For the experiments, the automatic metrics have the following input parameters:
-	Bilingual Evaluation Understudy Score (BLEU) – only one reference was used, the maximum n-gram length was set to four and no pre-processing techniques were applied.
-	Recall-Oriented Understudy for Gisting Evaluation (ROUGE) – the best values was achieved by measuring the match-rate of unigrams (ROUGE-1).
-	Bilingual Evaluation Understudy with Representations from Transformers (BLEURT) – a BLEURT checkpoint was used – a self-contained folder that contained a regression model which was tested on several languages, but should work for the 100+ languages of multilingual C4 (a cleaned version of Common Crawl’s web crawl corpus), including the Romanian language with 45M of training and 45K of validation examples. Specifically, BLEURT-20  was used as checkpoint, being a 32 layers pre-trained transformer model, named RemBERT, which contained 579M parameters fine-tuned on human ratings and synthetic data (~590K sentence pairs) collected during years 2015 and 2019 from WMT Metrics Shared Task.
-	BERTScore – “bert-base-multilingual-cased” was used as model type, a 12-layer transformer with token embeddings of size 768, trained by Google on the Wikipedia dumps from 104 languages, including Romanian which was explicitly added (e.g., lang=“ro”) and the selected number of layers was 9 (e.g., num_layers=9).

**Table 3 Scores of generated news items using RoGPT-2 and MCBGPT-2 models for the test and validation dataset**
|       Model name     |       Dataset      |    BLEU   |  ROUGE  | BLEURT | BERTScore | 
| -------------------- |:------------------:|---------- |:-------:| ------ |:---------:| 
|       RoGPT-2        |       Test         |    32.29  |   0.53  |  0.68  |  0.8106   | 
|                      |    Validation      |    28.70  |   0.50  |  0.52  |  0.8136   | 
|       MCBGPT-2       |       Test         |    8.79   |   0.25  |  0.63  |  0.8124   | 
|                      |    Validation      |    9.11   |   0.14  |  0.50  |  0.8277   | 

<img alt="Figure 4" title="Figure 4 - Distribution of unique words for RoGPT-2 and MCBGPT-2 models" src="https://user-images.githubusercontent.com/114503442/192863836-47bd9596-7812-4ba2-a3c1-03b888c3f585.png" width="200" height="200"> <img alt="Figure 5" title="Figure 5 - BLEU metric values of generated news items (30 values) using RoGPT-2 and MCBGPT-2 models" src="https://user-images.githubusercontent.com/114503442/192864035-fdbc86aa-8c99-4101-baa4-ac9273787b70.png" width="200" height="200"> <img alt="Figure 6" title="Figure 6 - ROUGE (red), BLEURT (green) and BERTScore (gray) metrics values (30 values) of generated news item" src="https://user-images.githubusercontent.com/114503442/192864204-a9cb3034-cb84-403f-8d8e-ba478d2a356f.png" width="200" height="200"> <img alt="Figure 7" title="Figure 7 - Distribution of ROUGE (red), BLEURT (green) and BERTScore (gray) metrics values of generated news items" src="https://user-images.githubusercontent.com/114503442/192864454-09d95d3a-4351-411c-9e0c-13b5a1440bd0.png" width="200" height="200">

This paper was focused on the quantitative evaluation (Figure 4, Figure 5, Figure 6, Figure 7) using different automatic metrics, and future studies will include only qualitative evaluation


In this research, a new GPT-2 architecture have been tested and evaluated, generating news items based on short Romanian text prompts. Using automatic metrics such as BLEU, ROUGE, BLEURT and BERTScore, the MCBGPT-2 and RoGPT-2 models are compared, thus, providing another solution for Romanian text generation systems. 
The MCBGPT-2 model achieves slightly better scores than RoGPT-2 model for the BERTScore metric when are evaluated larger sentences, considering the quantitative evaluation. 

Next, the code and examples.
