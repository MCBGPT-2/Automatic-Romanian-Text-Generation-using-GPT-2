# Automatic Romanian Text Generation using GPT-2

One of the most significant tasks in natural language processing (NLP) is text generation, which beneficiate from the recent architectures that use large pre-trained transformer models, such as the Generative Pre-trained Transformer-2 ([GPT-2](https://github.com/openai/gpt-2)) or GPT-3 developed by OpenAI, and Google’s Bidirectional Encoder Representations from Transformers ([BERT](https://aclanthology.org/N19-1423/)). The paper presents a NLG model based on the GPT-2 architecture that generates Romanian online news, using manually annotated texts. A small Romanian GPT-2 model, using 24 thousand news items, named MCBGPT-2 was developed, tested and evaluated. Additionally, an existing Romanian GPT-2 model, called [RoGPT-2](https://huggingface.co/readerbench/RoGPT2-base), was added to experiments. For evaluation, is presented a comparison of several automatic metrics such as [BLEU](https://dl.acm.org/doi/10.3115/1073083.1073135), [ROUGE](https://aclanthology.org/W04-1013), [BLEURT](https://aclanthology.org/2020.acl-main.704/) and [BERTScore](https://arxiv.org/abs/1904.09675) applied to generated news items from the test and validation datasets. Experimental results revealed that the MCBGPT-2 and RoGPT-2 models provided similar performances in text generation task for Romanian language, using less data for MCBGPT-2 model’s training process. 

**Dataset description**
---
Figure 1 presents the top news sources and the distribution of number of articles from each source, such as agerpres.ro – 2320, monitorulapararii.ro – 720, news.ro – 614 or evz.ro – 536 articles.
The aforementioned dataset was split into [train](https://github.com/MCBGPT-2/Automatic-Romanian-Text-Generation-using-GPT-2/blob/main/dataset/gpt_training_dataset.csv) (60%), [validation](https://github.com/MCBGPT-2/Automatic-Romanian-Text-Generation-using-GPT-2/blob/main/dataset/gpt_validation_dataset.csv) (20%), and [test](https://github.com/MCBGPT-2/Automatic-Romanian-Text-Generation-using-GPT-2/blob/main/dataset/gpt_test_dataset.csv) (20%).  The training process of the MCBGPT-2 model used 14,756 news items, while 4922 news items in each of the test and validation datasets were presented. 

Table 1 and Figure 2 presents the distributions of the test, validation, and train datasets, revealing that our datasets contain the same average words per news item (e.g., ~200 words) and the vocabulary size (unique words from The Explanatory Dictionary of the Romanian Language - DEX) of the test and validation datasets are well balanced (e.g., ~1M words), while the training dataset is about three times larger (e.g., ~3M words).

<div align="center">

<img alt="Figure 1" title="Figure 1 - Distribution of articles across news sources" src="https://user-images.githubusercontent.com/114503442/192862531-be93a66a-c36e-4479-aa99-c543d06d9a92.png" width="200" height="200"> <img alt="Figure 2" title="Figure 2 - Distribution of words across test, validation and training dataset" src="https://user-images.githubusercontent.com/114503442/192862554-41753d5c-d8d2-4f22-9378-3a2ae9a466f4.png" width="200" height="200">
</div>

<div align="center">

**Table 1 Distribution of words across test, validation and train datasets**
|        Words                  |   Dataset   |     No     |        Words                  |   Dataset   |     No     | 
| ----------------------------- |:-----------:|:----------:| ----------------------------- |:-----------:|:----------:| 
|                               |    Test     | 957,787    |                               |    Test     |    194     |
|    Romanian unique words      | Validation  | 1,026,680  |  Average words  per news item | Validation  |    208     |
|                               |   Train     | 2,837,236  |                               |    Train    |    192     |

</div>

The experiments were performed on a single server using tf.distribute.Mirrored Strategy, a TensorFlow API to distribute training across multiple GPUs and several software or packages such as CUDA (vers. 11.2), Python (vers. 3.9.7) and TensorFlow (vers. 2.4.1).The proposed MCBGPT-2 model was trained and validated using the aforementioned news corpus, that contain 18 fake news and 24,582 true news, 368 news with negative polarity and 2135 news with positive polarity, being split in equal parts between train, test and validation datasets. For this research, the Adam optimizer is used with a small learning rate (e.g., 3 x 105), a loss function such as sparse categorical cross-entropy and a vocabulary size of 60,000 words. Several training statistics of the proposed model, including the hyperparameters and training time, are presented in Table 2. 

<div align="center">

**Table 2 Training statistics of MCBGPT-2 model**
|    Parameters name   | Value of parameter | 
| -------------------- |:------------------:| 
| Number of parameters |       131M         | 
|    Number of epoch   |        15          |  
| Duration of an epoch |        5h          | 
|     Context size     |       512          | 
|      Batch size      |        12          | 

</div>

**The MCBGPT-2 model**
---
```python
block_size = 512
BATCH_SIZE = 12
BUFFER_SIZE = 1000
# defining our optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
# definining our loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# defining our metric which we want to observe
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
# compiling the model
model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])
num_epoch = 15
history = model.fit(dataset, epochs=num_epoch)
from transformers import WEIGHTS_NAME, CONFIG_NAME
output_dir = configuration.get_property('PYTHON_DIR') + "/gpt_models/mcb_model"
# creating directory if it is not present
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_to_save = model.module if hasattr(model, 'module') else model
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)
# save model and model configs
model.save_pretrained(output_dir)
model_to_save.config.to_json_file(output_config_file)
# save tokenizer
tokenizer.save_pretrained(output_dir)
```


**Generate text with the MCBGPT-2 model**
---
```python
output_dir = configuration.get_property('PYTHON_DIR') + "/gpt_models/mcb_model/"
CUDA_LAUNCH_BLOCKING = 1
# TensorFlow
from transformers import AutoTokenizer, TFAutoModelForCausalLM

results = sqlI.select_all_rows_from_table(['columns'], 'table', 'database', None, 'primaryKey DESC')
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = TFAutoModelForCausalLM.from_pretrained(output_dir)

updated_texts = dict()
for result in results:
    updated_texts.update({result[0]: configuration.clean_text(result[1])})

for id_single, text in updated_texts.items():
    max_length = 4000
    min_length = 600
    inputs = tokenizer.encode(text, return_tensors='tf', max_length=512, truncation=True)
    text_predicted = model.generate(inputs, max_length=int(max_length), min_length=int(min_length),
                                    no_repeat_ngram_size=2,
                                    temperature=0.8, num_beams=5, num_return_sequences=5, )
    new_text = tokenizer.decode(text_predicted[0], skip_special_tokens=True).replace('<|endoftext|>', '')
    new_text = new_text.replace("'", ' ')
    sqlI.update_field_for_all_rows_from_table(['column'], "'" + new_text + "'",
                                              'table',
                                              'primaryKey="' + str(id_single) + '" AND primaryKey > 0',
                                              'database').
```

**Experiments**
---
For the experiments, the automatic metrics have the following input parameters:
-	[_Bilingual Evaluation Understudy Score (BLEU)_](https://dl.acm.org/doi/10.3115/1073083.1073135) – only one reference was used, the maximum n-gram length was set to four and no pre-processing techniques were applied.
-	[_Recall-Oriented Understudy for Gisting Evaluation (ROUGE)_](https://aclanthology.org/W04-1013) – the best values was achieved by measuring the match-rate of unigrams (ROUGE-1).
-	[_Bilingual Evaluation Understudy with Representations from Transformers (BLEURT)_](https://aclanthology.org/2020.acl-main.704/) – a BLEURT checkpoint was used – a self-contained folder that contained a regression model which was tested on several languages, but should work for the 100+ languages of multilingual C4 (a cleaned version of Common Crawl’s web crawl corpus), including the Romanian language with 45M of training and 45K of validation examples. Specifically, BLEURT-20  was used as checkpoint, being a 32 layers pre-trained transformer model, named RemBERT, which contained 579M parameters fine-tuned on human ratings and synthetic data (~590K sentence pairs) collected during years 2015 and 2019 from WMT Metrics Shared Task.
-	[_BERTScore_](https://arxiv.org/abs/1904.09675) – “bert-base-multilingual-cased” was used as model type, a 12-layer transformer with token embeddings of size 768, trained by Google on the Wikipedia dumps from 104 languages, including Romanian which was explicitly added (e.g., lang=“ro”) and the selected number of layers was 9 (e.g., num_layers=9).

**Additional metrics used in our experiments ([See dataset](https://github.com/MCBGPT-2/Automatic-Romanian-Text-Generation-using-GPT-2/tree/main/dataset))**
---
[METEOR (Metric for Evaluation of Translation with Explicit ORdering)](METEOR (Metric for Evaluation of Translation with Explicit ORdering))

[GLEU metric (General Language Evaluation Understanding)](https://web.science.mq.edu.au/~rdale/publications/papers/2007/gleu4ps2pdf.pdf)


**BERTScore metric**
---
```python
output_dir = configuration.get_property('PYTHON_DIR') + "/gpt_models/bert-base-multilingual-cased/"
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelWithLMHead.from_pretrained(output_dir)

results = sqlI.select_all_rows_from_table(['sql columns'],
                                          'table',
                                          'database', None, 'primaryKey ASC')

stop = stopwords.words('romanian')
if os.path.isabs(configuration.get_property('CONFIG_PATH') + os.sep + 'stop_words.txt'):
    with open(configuration.get_property('CONFIG_PATH') + os.sep + 'stop_words.txt', 'r') as opened_file:
        lines = opened_file.read().split("\n")
        for line in lines:
            if configuration.clean_text(line) not in stop:
                stop.append(configuration.clean_text(line))
        opened_file.close()

for id_single, text, new_text in results:
    try:
        P, R, F1 = score([text], [new_text], model_type=output_dir, num_layers=9, lang="ro", verbose=True)
        final = F1.mean()
        value = final.item()
        sqlI.update_field_for_all_rows_from_table(['sql columns'], value,
                                                  'table',
                                                  'primaryKey="' + str(id_single) + '" AND primaryKey > 0',
                                                  'database')
```

<div align="center">

**Table 3 Scores of generated news items using RoGPT-2 and MCBGPT-2 models for the test and validation dataset**
|       Model name     |       Dataset      |    BLEU   |  ROUGE  | BLEURT | BERTScore | 
|:--------------------:|:------------------:|---------- |:-------:| ------ |:---------:| 
|       RoGPT-2        |       Test         |    32.29  |   0.53  |  0.68  |  0.8106   | 
|                      |    Validation      |    28.70  |   0.50  |  0.52  |  0.8136   | 
|       MCBGPT-2       |       Test         |    8.79   |   0.25  |  0.63  |  0.8124   | 
|                      |    Validation      |    9.11   |   0.14  |  0.50  |  0.8277   | 

</div>

<div align="center">
    
<img alt="Figure 4" title="Figure 4 - Distribution of unique words for RoGPT-2 and MCBGPT-2 models" src="https://user-images.githubusercontent.com/114503442/192863836-47bd9596-7812-4ba2-a3c1-03b888c3f585.png" width="200" height="200"> <img alt="Figure 5" title="Figure 5 - BLEU metric values of generated news items (30 values) using RoGPT-2 and MCBGPT-2 models" src="https://user-images.githubusercontent.com/114503442/192864035-fdbc86aa-8c99-4101-baa4-ac9273787b70.png" width="200" height="200"> <img alt="Figure 6" title="Figure 6 - ROUGE (red), BLEURT (green) and BERTScore (gray) metrics values (30 values) of generated news item" src="https://user-images.githubusercontent.com/114503442/192864204-a9cb3034-cb84-403f-8d8e-ba478d2a356f.png" width="200" height="200"> <img alt="Figure 7" title="Figure 7 - Distribution of ROUGE (red), BLEURT (green) and BERTScore (gray) metrics values of generated news items" src="https://user-images.githubusercontent.com/114503442/192864454-09d95d3a-4351-411c-9e0c-13b5a1440bd0.png" width="200" height="200">
</div>


This paper was focused on the quantitative evaluation (Figure 4, Figure 5, Figure 6, Figure 7) using different automatic metrics, and future studies will include only qualitative evaluation


In this research, a new GPT-2 architecture have been tested and evaluated, generating news items based on short Romanian text prompts. Using automatic metrics such as BLEU, ROUGE, BLEURT and BERTScore, the MCBGPT-2 and RoGPT-2 models are compared, thus, providing another solution for Romanian text generation systems. 


The MCBGPT-2 model achieves slightly better scores than RoGPT-2 model for the BERTScore metric when are evaluated larger sentences (Table 4), considering the quantitative evaluation. 

**Table 4 Examples of Romanian generated sentences**

<table align="center">
  <tr>
    <td align="center">State</td>
    <td align="center" colspan="4">Text</td>
  </tr>
  <tr>
    <td align="center">Original</td>
    <td align="center" colspan="4">Coordonatorul campaniei naționale de vaccinare, medicul militar Valeriu Gheorghiță a vorbit despre finalul pandemiei în România.Întrebat, într-o  emisiune televizată, când crede că se va reveni la viața normală, doctorul Gheorghiță a spus Eu sper căt mai repede. Dacă mă întrebați de o lună anume, cred că undeva în iulie august ușor ușor vom intra într-o normalitate care va fi din ce în ce mai concretizată pe măsură ce nivelul de acoperire vaccinală va crește undeva peste a spus medicul Valeriu Gheorghiță...</td>
  </tr>
    <tr>
    <td align="center">Link</td>
    <td align="center" colspan="4">https://www.ziarulprofit.ro/index.php/cand-se-va-termina-pandemia-in-romania-doctorul-valeriu-gheorghita-da-raspunsul-video/</td>
  </tr>
  <tr>
    <td align="center" colspan="5">Generated text</td>
  </tr>
  <tr>
    <td align="center">RoGPT-2 model</td>
    <td align="center" colspan="4">….a spus medicul Valeriu Gheorghiță, care a mai spus că s-a întâlnit cu ministrul sănătății, Sorina Pintea, pentru a discuta despre starea de sănătate a populației și despre măsurile care se vor lua pentru ca să nu se mai ajungă la astfel de situații, iar ministrul a promis că va avea o discuție cu el și va vedea care sunt măsurile pe care ministerul le va lua, dar și ce măsuri vor fi luate de către minister, atât pentru prevenirea, cât și pentru tratarea celor care au fost depistați cu COVID-19 și care nu au putut fi vaccinați, precum și măsuri de combatere a epidemiei de coronavirus și a altor boli care pot fi transmise prin contact direct cu o persoană infectată cu virusul imunodeficienței umane (HIV).</td>
  </tr>
    <tr>
    <td align="center">Metrics</td>
        <td align="center">BLEU</td><td align="center">ROUGE</td><td align="center">BLEURT</td><td align="center">BERTScore</td>
  </tr>
    <tr>
    <td align="center">Scores</td>
        <td align="center">22.43</td><td align="center">0.39</td><td align="center">0.83</td><td align="center">0.8487</td>
  </tr>
    <tr>
    <td align="center">MCBGPT-2 model</td>
    <td align="center" colspan="4">….a spus medicul Valeriu Gheorghiță. În ceea ce privește vaccinarea cu cea de-a treia doză de vaccin, medicul a explicat că, într-adevăr, nu se pune problema de a face o formă gravă de boală, dar nu este un lucru foarte important pentru noi toți cei care se vaccinează cu prima schemă cu vaccinul de la compania Johnson  Johnson, pentru că nu avem niciun fel de imunitate de grup peste 40%, iar pentru persoanele care au fost vaccinate cu schemă completă au un risc scăzut de cel puțin 20 de ori să se infecteze și cu o creștere a numărului de persoane care s-au vaccinat cu schema completă. De asemenea, el a mai spus că este important să avem o rată de pozitivare a testelor de maxim 20 la sută, a adăugat medicul.</td>
  </tr>
     <tr>
    <td align="center">Metrics</td>
        <td align="center">BLEU</td><td align="center">ROUGE</td><td align="center">BLEURT</td><td align="center">BERTScore</td>
  </tr>
    <tr>
    <td align="center">Scores</td>
        <td align="center">20.74</td><td align="center">0.38</td><td align="center">0.75</td><td align="center">0.8588</td>
  </tr>
</table>


**Articles**
---
[Automatic Fake News Detection for Romanian Online News](https://www.mdpi.com/2078-2489/13/3/151)

[Automatic Irony Detection for Romanian Online News](https://ieeexplore.ieee.org/document/9259715)

[Targeted Romanian Online News in a Mobile Application Using AI](http://rochi.utcluj.ro/articole/8/RoCHI2020-Buzea.pdf)

[A Three Word-Level Approach Used in Machine Learning for Romanian Sentiment Analysis](https://ieeexplore.ieee.org/document/8909458)

**Thanks**
---
I am really thankful to have you as both professor and academic advisor - [Ștefan TRĂUȘAN-MATU](https://scholar.google.com/citations?hl=ro&user=p_KpBToAAAAJ) and [Traian REBEDEA](https://scholar.google.com/citations?hl=ro&user=7NxaE1MAAAAJ)
