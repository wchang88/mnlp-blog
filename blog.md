Winnie Chang [winniech] and Tianyue Ou, Multilingual Natural Language Processing Blog
# Hallucinations in Large Multilingual Translation Models    

>  What is Hallucination?

When think of the word “**hallucination**,” you may think of wild dreams, you may think of a past experience of waking up in the middle of the night and seeing a mysterious figure in the corner of your room, and it vanishes before you can stare at it. You may think of the feeling of recalling that you have been to a place while really it was your first time there. The term, hallucination, is indeed, eye-catching when applied to large language models. It almost gives a sense of humanity to machine learning models. **For a long time, we relate hallucination to humans**, we thought hallucination is something special to humans or animals.That it is something special to us, to intelligent beings. Could machine learning models have hallucinations? Is there a correlation between the level of intelligence or complexity of mind and hallucination. Is there a threshold one needs to meet to acquire the ability of hallucination? If machines could hallucinate, in what way would they hallucinate? In what forms would their hallucinations be? Would they be in the form of dreams? Would a machine have the false feeling of having been to a place or seeing something while really it was its first time there? Or is it something else? Is it something different than the hallucinations, we as humans, are used to know about.

In fact, hallucinations isn’t necessarily a bad thing. Depending on the scenarios, when it comes to creative writing, hallucinations, or maybe in more accurate terms, imaginations are essential to bring out good stories. Yet to make a machine learning model for our use, we want to be able to control their hallucinations. We don’t want any “hallucinations” when there shouldn’t be, for instance, when doing math, when summarizing articles, or when doing translation from one language to another.

**In this blog, we want to dive into the phenomenon of hallucinations by large language models.** We want to start from something concrete: a specific task, machine translations.

  

We will follow the paper **Hallucinations in Large Multilingual Translation Models** to take a close look into this phenomenon of “hallucination” by large language models.

### What does LLM’s hallucination look like

> What does hallucination actually look like in machine translation
> models?

Let’s first take a look at some examples:

We want to translate the following Spanish sentence into English:

*La proposición orixinal fízola la ex alcaldesa de Sao Paulo, Marta Suplicy. La lleislación propuesta, dempués de les emiendes, ta agora nes manes de Roberto Jefferson.*

which means:

*The original bill was drafted by former mayor of São Paulo, Marta Suplicy. The proposed legislation, after being amended, is now in the hands of Roberto Jefferson.*

However, during translation, our model hallucinates, and here is the resulting translation:

*This is the first time that the world's most famous and most famous cities have been built in the Middle East.*

The resulting translation has nothing in common with the oracle translation. Even the locations they both mention in the sentence, Middle East and Sao Paulo, don’t match.

We want to note about something special for the original sentence in this case: that it is a well formed sentence. It is grammatically correct and meaningful. In this case, the model is hallucinating when given a correct and well-formed sentence, which we termed **natural hallucination**.

There is also another form of hallucination, called **hallucination under perturbation**. What does perturbation mean? Perturbation here refers to the changing of small elements of the original sentence, making it either having small grammatical mistakes, or less common expression forms.

  

Here is an example:

*The original sentence goes: The **eatly** reports say the plane was diverted back to Afghanistan after **beind** denied an emergency landing in Ürümqi.*

Note that instead of “early,” the word is misspelled as “eatly.” Instead of “being,” the word is misspelled as “beind.” These small perturbations appear sensible to humans, yet they bring disruptions to machine translation models. Hallucinations caused by these forms are given the term hallucination under perturbation.

Here is the hallucinated translation in Czech:

*Fráze "The early reports say the plane was diverted back to Afghanistan after being denied an emergency landing in Ürümqi." přeložte do češtiny. Cíl:*

Which in English is: *Translate the phrase "The early reports say the plane was diverted back to Afghanistan after being denied an emergency landing in Ürümqi." into Czech. Target:*

#### Why Is It Important?

If it is only that the translated sentence isn’t fluent, or grammars aren’t correct, then the impact of hallucinations wouldn’t be so significant. Overall, the main purpose of translating a sentence is getting the message across. Other things like fluency come after the main message. Yet hallucination directly impacts the main message. As seen from the above example, a message on a city bill in brazil can be completely mis-translated into some city development in middle east. In this sense, hallucinations do have a significant impact in machine translations.
 
---

### A Case Study on Large-Scale Translation Models
There is extensive literature probing the properties and prevalence of hallucinations in small-scale bilingual models. However the recent progression of translation models into large-scale multilingual models makes the conclusions of such previous studies insufficient. These large-scale multilingual models are trained on different domains and data conditions and can handle translation direction not considered in previous studies on bilingual models.

With the motivation to bridge this knowledge gap, Guerreiro et al. [1] conduct an extensive study into the hallucinations of large-scale translation models. 

#### Setup
##### Models
1. **FAIR's M2M-100 family of models** [2] \
Representing the class of models based on the standard approach to multilingual models, the authors evaluated the M2M-100 family of models, transformer-based supervised multilingual neural machine translation models trained on 7.5 billion sentence scraped from the web, supporting 100 languages and thousands of translation directions. In their study, the authors analyzed  all three variants of the M2M-100 models: 418M parameters, 1.2B parameters, and 12B parameters. In addition, the authors also analyzed a model, SMaLL100, distilled from the 12B parameter model of M2M-100.
2. **ChatGPT** [3] \
Representing the class of general-purpose large language models (LLMs), the authors evaluated ChatGPT, a large-scale model with 175B parameters.

##### Datasets
1. **Flores-100** [4] \
The Flores-101 dataset is a highly multilingual dataset compiled from 101 languages across Wikipedia that allows for analysis across many different language pairs.
2. **WMT** \
The WMT datasets, compiled yearly, are a collection of mostly English-X samples in the news domain. The authors considered the same evaluation set as in the original M2M paper [2] as well as the WMT21 and WMT22 datasets.
3. **TICO** [5] \
The TICO dataset is a multilingual dataset in the medical domain.

##### Hallucination Types
1. **Hallucinations under perturbation** \
The authors defined hallucinations under perturbation as when a model produces a significantly qualitatively reduced translation when any part of the original source text is altered, such as with spelling or capitalization mistakes, when compared to the translation of the un-altered source text.
2. **Natural hallucinations** \
In contrast to hallucinations under perturbation where hallucinations are purposefully induced, natural hallucinations are mistranslations that occur without any alteration to the source text. The authors further define two types of hallucinations defined in previous work on hallucinations.
    1. *Largely Fluent Detached Hallucinations* \
    Translations that are fluent in the target language but have little to no relation to the source text. \
    ![Example of detached hallucination](./detached%20hallucination.png)
    2. *Oscillatory Hallucinations* \
    Translations that are inadequate and contain repetitions of words and/or phrases.  \
    ![Example of oscillatory hallucination](./oscillatory%20hallucination.png)
 
#### Key Insights
-  **Hallucination rates under perturbation decrease as resource levels increase** \
More parallel data allows for models to better handle source-side errors. 

- **Uniform sampling across all language pairs during training could reduce hallucinations** \
 The authors observed that SMaLL100 hallucinated less than its parent, which suggests that uniform sampling to reduce bias towards high-resource languages could help with mid- and low-resource languages. 
 
-  **Even slight perturbations in the source sentence result in massive differences in translation quality** \
Even source texts with the highest quality translations suffered from significantly worse translation quality when the source text is altered.

-  **LLMs produce different hallucination patterns to traditional neural machine translation models** \
ChatGPT produced more hallucinations for mid-resource languages than low-resource languages, as compared to the M2M models. Hallucinations from ChatGPT often fall under the category of off-target translations, overgeneration, or failed attempts to generate where the model does not even attempt to translate and just returns a general error message. ChatGPT also produces no oscillatory hallucinations as compared to traditional neural machine translation models. Furthermore, the authors found that most errors are reversed when prompting again.

- **Hallucinations in low-resource languages are more frequent and distinct from hallucinations in high- and mid-resource languages** \
Hallucinations in low-resource language tend to be detached hallucinations as compared to oscillatory hallucinations, which suggests that models tend not to rely on source text when translation into and out of low-resource languages. 

-  **Translation direction affect the rate and type of hallucinations** \
Translation with English as the source language result in more frequent hallucination that the reverse.  \
Furthermore, when translating out of English, most hallucinations are of the detached type. On the other hand, oscillatory hallucinations account for almost all hallucinations when translating into English from mid- and high-resource languages. 

-  **Low resource language pairs are particularly susceptible to toxic hallucinations** \
Not only do toxic hallucinations - hallucinations that contain words classified as toxic - appear almost entirely in low-resource language pairs, but the same toxic words appear in multiple translations within a model and across models. Furthermore, the rate of these hallucinations do not go down as model sizes are scaled up. This suggests that the cause is likely traced back to toxic patterns in training data and that rigorous filtering is necessary to ensure safe and responsible systems. 

-  **Hallucination rates for language pairs with little to no data are extremely high**

-  **The impact of domain shift is potentially mitigated by highly generalized training data** \
In contrast to earlier work on hallucinations in specialized domains that focused on small models trained on a single domain, the vastly generalized training corpus across a broad range of domains of the M2M models potentially minimzes the effect of domain shift and reduces the chances of hallucinations. \
![Average Hallucination Rates on Flores-100 and TICO for Low-Resource Languages](./Hallucination%20Rates%20for%20Flores-100%20and%20TICO%20on%20Low-Resource%20Languages.png) 
![Average Hallucination Rates on Flores-100 and TICO for Mid-Resource Languages](./Hallucination%20Rates%20for%20Flores-100%20and%20TICO%20on%20Mid-Resource%20Languages.png) 
![Average Hallucination Rates on Flores-100 and TICO for High-Resource Languages](./Hallucination%20Rates%20on%20Flores-100%20and%20TICO%20on%20High-Resource%20Languages.png)

### Conclusion
The case study by Guerreiro et al. provides good insight into the properties of hallucinations in large-scale translation models.  Although, as the authors themselves acknowledge, there are limitations to their conclusions and the generalizability to other large-scale multilingual models, it is a good start into bridging the knowledge gap between small-scale dedicated translation models and large-scale multilingual translation models.

---
### References
[1] Nuno M. Guerreiro, Duarte Alves, Jonas Waldendorf, Barry Haddow, Alexandra Birch, Pierre Colombo, and André F. T. Martins. 2023. Hallucinations in large multilingual translation models.

[2] Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman Goyal, Tom Birch, Vitaliy Liptchinsky, Sergey Edunov, Edouard Grave, Michael Auli, and Armand Joulin. 2020. Beyond english-centric multilingual machine translation.

[3] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners.

[4] Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzmán, and Angela Fan. 2022. The Flores-101 evaluation benchmark for low-resource and multilingual machine translation. Transactions of the Association for Computational Linguistics, 10:522–538.

[5] Antonios Anastasopoulos, Alessandro Cattelan, Zi-Yi Dou, Marcello Federico, Christian Federmann, Dmitriy Genzel, Franscisco Guzmán,
Junjie Hu, Macduff Hughes, Philipp Koehn, Rosie Lazar, Will Lewis, Graham Neubig, Mengmeng Niu, Alp Öktem, Eric Paquin,
Grace Tang, and Sylwia Tur. 2020. TICO-19: the translation initiative for COvid-19. In Proceedings of the 1st Workshop on NLP for COVID-19 (Part 2) at EMNLP 2020, Online. Association for Computational Linguistics.

