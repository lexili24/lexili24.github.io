---
title: "Meta-Learning for NLU tasks under Continual Learning Framework"
#published: false
---

Co-authors: Yong Fan, Duo Jiang, Shiqing Li, Jiacheng Wang 



## Brief summary 
Neural network has been recognized with its accomplishments on tackling various natural language understanding (NLU) tasks. Methods have been developed to train a robust model to handle multiple tasks to gain a general representation of text. In this paper, we implement the model-agnostic meta-learning (MAML) and Online aware Meta-learning (OML) meta-objective under the continual framework for NLU tasks proposed by [Javed and White](https://arxiv.org/pdf/1905.12588.pdf). We validate our methods on selected SuperGLUE and GLUE benchmarks.

## Introduction

One ultimate goal of language modelling is to construct a model like human, to grasp general, flexible and robust meaning in language. One reflection of obtaining such model is be able to master new tasks or domains on same task quickly. However, NLU models have been building from specific task on given data domain but fail when dealing with out-of-domain data or performing on a new task. To combat this issue, several research areas in transfer learning including domain adaptation, cross lingual learning, multi-task learning and sequential transfer learning have been developed to extend model handling on multiple tasks. However, transfer learning tends to favor high-resources tasks if not trained carefully, and it is also computationally expensive.Meta-learning algorithm tries to solve this problem by training model in a variety of tasks which equip the model the ability to adapt to new tasks with few samples.

In this post, I will walk you through how we adopt the idea of model-agnostic meta learning (MAML) which is an optimization method of meta-learning that directly optimized the model by constructing an useful initial representation that could be efficiently trained to perform well on various tasks. However, in an continual learning where data comes into the model sequentially, there is a potential problem of catastrophic forgetting where a model trained with new tasks starts to perform worse on previously trained tasks. Therefore, there are two objectives of designing a meta-learning with continual learning framework: 1) accelerate future learning where it exploits existing knowledge of a task using general knowledge gained previous tasks 2) avoid interference in previous tasks by updates from new tasks. 


## Background

Before we dive into the implementation details, lets take some time to examine meta learning and continual learning in natural language and other fields, which leads us to develop our framework tackling NLU tasks. Plenty of research have been focused in these two areas and some efforts have succeeded in combining these two goals in other field. 

### Meta-Learning

There has been success in implementing MAML in NLU tasks. [Dou's](https://arxiv.org/pdf/1908.10423.pdf) work explored the model-agnostic meta-learning algorithm (MAML) and its variants for low-resource NLU tasks and obtained impressive results on the GLUE benchmark. In addition, meta learning is proved to excel in other natural language domains. [Mi](https://arxiv.org/pdf/1905.05644.pdf) has shown promising results of incorporating MAML in natural language generation (NLG). NLG models, like many NLU tasks, are heavily affected by the domain they are trained on and are data-intensive but data resource is low due to high annotation cost. Mi's approach to generalize a NLG model with MAML to train on the optimization procedure and derive a meaningful initialization serves to adapt new low-resource NLG scenarios efficiently. In comparison, Meta-Learning approach outperformed multi-task approach with higher BLEU score and lower error percentage. 

### Continual Learning

Continual learning is proved to boost model performance in [Liu's paper](https://arxiv.org/pdf/1904.09187.pdf). Liu leveraged continual learning to construct a simple linear sentence encoder to learn representations and compute similarities between sentences, such application can be fed into a chat bot. A general concern is that in practice, the encoder is fed into a series of input from inconsistent corpora, and might degrade performance if fails to generalize common knowledge across domains. Continual learning enables zero-shot learning and allows a sentence encoder to perform well on new text domains while avoiding catastrophic forgetting. Authors evaluate result on semantic textual similarity (STS) datasets with Pearson correlation coefficient (PCC). With a structure utilizing continual learning approach, Liu showed consistent results cross various corpora. 

With sunny days there's also rainy days. Continual learning implemented in NLU tasks on top of transfer learning by [Yogatama's ppaer](https://arxiv.org/pdf/1901.11373.pdf) did not show generalization of the model. Yogatama followed continual learning setup to train a new task on best SQuAD-trained BERT and ELMo model, and both architectures show catastrophic forgetting after TriviaQA or MNLI is trained, which degrades model performance on SQuAD dataset. Their work shows an attempt to derive a generative language model and provides a solid ground of continual learning in language modelling. 

Let's quickly jump outside of NLP field. An implementation of meta-learning under continual framework is proposed in reinforcement learning (RL) by [Alshedivat](https://arxiv.org/pdf/1710.03641.pdf). In their paper, MAML is proved to be a complementary solution adding onto continual adaption in reinforcement learning (RL) fields. Al-Shedivat considered nonstationary environments as sequences of stationary tasks for RL agents, which transferred nonstationary environment to learning-to-learning tasks. They developed a gradient-based Meta-Learning algorithm for quick adaption to continuously changing environment. They found that Meta-Learning is capable of adapting far more efficiently than baseline models in the few-shot regime. Although the implementation is outside the domain of Natural Language Processing, it is worth-noting that experts from different domains have implemented this method and sheds lights on authors to implement in NLU tasks. 

## Method
### Problem Formation
We followed Dou's defintion on splitting Meta-training and Meta-testing tasks. We use high resources tasks for Meta-Training, namely SST-2, QQP,1 MNLI and QNLI. We use low-resource auxiliary tasks in Meta-testing, other than CoLA , MRPC , STS-B and RTE from Glue, we extend it on SuperGlue low-resource tasks: RTE, BoolQ, CB, Copa, WiC and WsC.

### Method
Javed and White proposed a methodology that achieves Meta-Learning under continual learning setting. The representation learnt from existing knowledge by Meta-Learning, enables the model to learn new tasks quickly. Traditional MAML, proposed by [Finn](https://arxiv.org/pdf/1703.03400.pdf), during meta-traning, a task T_i is sampled and we train the model with K samples, collect feedback to task's loss function, and evaluate model on new samples selected from the same task. Model is improved by looking at how test error on new data changes with respect to parameters. Finn's approach of MAML is to learn an effective initialization which Javed and White reframed to MAML-Rep, a MAML like architecture that is tailored for online setting. OML is another approach that attempts to alleviates catastrophic forgetting by online updating at meta-training phase, and utilize meta-testing tasks to improve the accuracy of the model in general. Given most neural networks are highly sparse, OML takes such advantage to update its parameters by constructing representations of the incoming online data point of different tasks either as parallel, where some updates can be beneficial for many tasks, or orthogonal, where updates by later task do not interfere with previous tasks. 

Our model architecture strictly follows the architecture proposed by Javed and White, where both MAML-Rep and OML objectives are trained and evaluated in NLU tasks by training a pre-trained BERT model, we call models produced by these objectives MAML-Bert and OML-Bert. We select pre-trained BERT as the backbone for our model.

For Meta-Training, we consider two Meta-Objective to minimize. (1) MAML-Rep and (2) OML objective. Both approach consists of two parts: a Representation learning network (RLN) and a Prediction learning network (PLN). RLN and PLN are seperated since they are trained independently. In our scope, ```bert-based-uncased``` is considered as RLN, and PLN is a simple classifier. MAML-Rep and OML both updates PLN only in Meta-Training inner loop using support dataset with around 100 data points for some inner steps, and updates RLN in Meta-Trianing using query dataset with 80 data points in the outer loop once. For Meta-Testing, 100 data points are fed into the model and both RLN and PLN is updated at some inner steps. At the end of each Meta-Tesing, we compare the performance of model trained on a specific task against model performance after seeing on all Meta-Testing Tasks, if model forgets we should see performance degrades on earlier tasks. Different from MAML-Rep, OML only trains with one data point at meta-training inner step. 


## Evaluation
To measure catastrophic forgetting problem, validation data from the meta-testing is applied to the model twice, once after the model trained on the specific task, and another after the model finishes training on all tasks. The sequence of tasks are unchanged for meta-testing, which follows the order of Cola, MRPC, STS-B and RTE. Therefore, since RTE is trained as the last task in the sequence, the result table only shows once score. For other tasks, there are two scores in the result table, first being evaluation right after model trained on that task, second being after model trained on all tasks. Note that Cola is evaluated with Matthew correlation, sts-b is evaluated with Pearson correlation, and rest two present accuracy score. 

We present our resuls along with Dou's paper and our baseline. Dou has only one score because it builds seperate model for specific meta-testing tasks individually after meta-training. Our baseline model is trained on 4 low-resource tasks only with 1000 data points from the training set, and evaluated with entire dev set. Similar to MAML-Rep and OML, there are two scores presented excluding RTE which is the last task in the sequence to train. 

<style>
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}
th, td {
  padding: 5px;
}
th {
  text-align: left;
}
</style>

<table style="width:100%">
  <tr>
    <th>Tasks</th>
    <th>CoLA</th> 
    <th>MRPC</th>
    <th>STS-B</th>
    <th>RTE</th>
  </tr>
  <tr>
    <td>Dou</td>
    <td>53.4</td> 
    <td>85.8</td>
    <td>87.3</td>
    <td>76.4</td>
  </tr>
  <tr>
    <td>BERT</td>
    <td>46.23/0</td> 
    <td>75/56.25</td>
    <td>65.28/68.56</td>
    <td>56.25</td>
  </tr>
  <tr>
    <td>MAML-Rep</td>
    <td>65.51/51.23</td>
    <td>86.27/88.0</td>
    <td>82.26/80.51</td>
    <td>90.25</td>
  </tr>
  <tr>
    <td>OML</td>
    <td>0/0</td>
    <td>68.3/68.3</td>
    <td>0/4.11</td>
    <td>52.7</td>
  </tr>
  <p>Table 1. Results of MAML-Rep and OML compared to Dou and baseline model. In Meta-Training, four high resource Glue tasks are randomly ordered. In Meta-Testing, low resource Glue tasks are trained in the same order as presented in the table.</p>
</table>


In addition, we swapped Meta-Testing Glue tasks to low resource SuperGlue tasks and expanded number of tasks to 5 showing in table 2. All tasks are evaluated with accuracy score. MAML-Rep outperforms on three out of four tasks, note that OML still struggles to get better than random guessing for WsC tasks, and behave like random guessing for WiC and Copa tasks. 


<table style="width:100%">
  <tr>
    <th>Tasks</th>
    <th>WsC</th> 
    <th>WiC</th>
    <th>BoolQ</th>
    <th>Copa</th>
    <th>Cb</th>
  </tr>
  <tr>
    <td>MAML-Rep</td>
    <td>75/72.11</td> 
    <td>52.98/53.29</td>
    <td>64.83/59.70</td>
    <td>55/55</td>
    <td>100</td>
  </tr>
  <tr>
    <td>OML</td>
    <td>36.53/36.53</td>
    <td>50/50</td>
    <td>61.82/62.17</td>
    <td>54.34/55</td>
    <td>89.12</td>
  </tr>
  <p>Table 2. Results of MAML-Rep and OML. In Meta-Training, four high resource Glue tasks are randomly ordered. In Meta-Testing, low resource SuperGlue tasks are trained in the same order as presented in the table.</p>
</table>



## Conclusion
In this work, we are able to extend Meta-Learning under continual learning framework to learn a general presentation that is robust on a set of continual tasks with efficiency. We replicate Javed and White's method and implement on NLU tasks. Results show that with less datapoints, we could derive a MAML like model that is robust on testing tasks, however extending it to continual setting during Meta-Training phrase, the performance drastically worsen. Future direction would be extending this approach to other language models, as wells as experiment with a combination of high and low resources other than Glue and SuperGlue benchmark, and unsupervised tasks to evaluate model performance. 

## Implementation Details
Our implementation is based on PyTorch implementation, backboned in Huggingface ``bert-base-uncased``` model. We use Adam optimizer, with a batch size of 16 for both Meta-Training and Meta-Testing. Maximum sentence length is set to be 64. 

In Meta-Learning and Meta-Testiing stage, we use learning rate of 5e^-5 for outer loop learning rate where we update RLN, and 5e^-3 for inner learning to update PLN. We use a cosine annealing in Meta-Training as a scheduler to update the optimizer. Dropout of 0.1 is applied to PLN when it is applicable. We set the inner update step to 5 for Meta-Training and 7 for Meta-Testing. We use a total sample of 128 and 112 for support and query dataset for Meta-Testing, and 100 and entire dev set during Meta-Testing to align with baseline results. 

Meta-Testing stage is invoked everytime trainting Meta-Training trained on 5 epochs. Meta-Training high resource Glue tasks, SST-2, QQP,1 MNLI and QNLI are randomly picked at each epoch. Meta-Testing low resource Glue tasks, CoLA , MRPC , STS-B and RTE and low resource SuperGlue tasks, RTE, BoolQ, CB, Copa, WiC and WsC are always trained in sequence to examine catastrophic forgetting problem.