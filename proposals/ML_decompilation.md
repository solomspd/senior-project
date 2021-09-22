# Machine learning decompilation

#Introduction
Although reverse engineering a binary program has a rather mature ecosystem of tools such as ghidra and binary ninja, they all still involve large amount of manual time and effort to bring back some sense of structure and human readability to the assembly code. This is mainly due to their dependence on pattern matching and other rule-based approaches which introduces many limitations to traditional decompilers including poor scalability and development. Since the purpose of decompilation is crucial to multiple cybersecurity domains such as malware analysis and vulnerability discovery, this process needs to be enhanced in order to offer better outcomes to the community.
Here, machine learning offers an opportunity to dramatically automate this process.


# Related Work
Machine learning based decompilation has been explored in a number of research projects before []. Each of these projects dealt with a different aspect that was not properly handled by commonly used decompilers such as variables' data type inference, function return and parameter type inference, or function entry point detection []. Escalada et al. focused on improving the accuracy of high-level function return types by building better classification models that increased the accuracy of the task[]. Chua et al., on the other hand, improved the detection of the parameters' number of a function along with detecting their types mainly through using building 4 different recurrent neural networks(RNN) for handling the tasks of parameters' type inference as well as recovering the types of caller arguments []. Their work achieved 81% parameter inference accuracy and 84% parameter counting accuracy. 

State-of-the-art techniques in this field focus on improving the process of decompilation as a whole through using neural machine translation (NMT) models. In this way, decompilation is dealt with as a translation problem between low level programming languages and high level ones. This results in massive improvements with regard to the clarity and accuracy of the generated high-level PL codes from the decompilers [,]. Despite how promising NMT-based decompilation approaches sound, multiple challenges are still prevalent with most of its previous implementations and projects.  

Most of the neural-based decompilation techniques are only able to recover accurate semantic for simple functions making them useless for any real-world code []. In addition, since the input data and output results of decompilation tools are highly structured (trees/ graphs) and with long-range dependencies, current NMT models are incompatible with them since they primarily work on sequential data []. Multiple projects were proposed to overcome these key challenges.

Katz et al. in 2019 proposed "Trafix", a decompiler based on Long Short Term Memory(LSTM) networks that offers a new way to preprocess the input assembly language and post-process the resulting high-level C code. This resulted in much improved syntax and symantic accuracy. However, it didn't perform well enough on decompiling conditional statements and loops [].


Later on, in 2021, Liang et al. managed to improve the existing neural decompilation techniques and proposed a complete framework for neural program decompilation, called "Neutron" []. Neutron is composed of three main phases to accomplish the decompilation task, code preprocessing, neural translation, and function reconstruction. Code preprocessing is mainly focused on the standardization of programming languages to help the model learn the association between low-level and high-level PL. In this phase, a neural-based decompilation model is trained to learn the conversion rules between the assembly code of an executable program and its functionally-equivalent high-level C code.  The actual translation happens in the neural translation phase using the model trained in the first phase. The final phase focuses on functions reconstruction to restore its structure. Neutron framework is assessed against real-world code projects achieving an average accuracy of 96.96% with regard to code readability and functionality recovering [].


Another very recent contribution in the area of neural-based decompilers is N-Bref framework []. It aims to solve multiple of the key challenges faced by previous attempts of neural-based decompilers. To solve the problem of incompatibility between the neural architectures of machine translation and the intrinsic structures of decompiler's data, a back-bone structural transformer was developed using inductive Graph Neural Networks (GNNs) to represent low-level code (LLC) as control/data flow dependency graphs (LLC Encoder) and source code as an Abstract Syntax Tree (AST Encoder) []. It additionally integrates an AST decoder as well as memory augmentation techniques to tackle the poor scalability problem that is faced with the growing size of programs. Since decompilation is composed of too many subtasks(eg, data type recovery and source code generation), Chen et al. avoided handling them all using one neural network, they instead composed the process into separate tasks using the same backbone structural transformer with different parameters to achieve better performance for each for these tasks. According to [], N-Bref framework indeed boosted the performance of decompilation and outperformed previous neural-based decompiler. 


# Contribution
A great number of the recent advances in decompilers have focused squarely on the desktop x86 architecure, here we intend to implement these more advanced methods to the android platform and its underlying byte-code.

`hile there has a been huge progress achieved with ML-based decompilation, there are many aspects that are not explored as well as key limitations that are still prevalent in the currently proposed ML-assisted decompilers. The most significant of these limitations is the decompilers' inability to reverse engineer obfuscated or optimized binary samples. However, since decompilers and similar tools are mostly used by malware analysts and other security professionals, these tools need to be able to deal with the highest degree of sophistication that might be present in executable programs, this is mainly due to the complexity of the binary samples offered by malware developers. Therefore, we propose a ML-based decompiler tool that is driven by the most popular techniques that are used by malware developers to hinder the efficiency or slow down the process of binary sample analysis.

Hence, our key contributions to this idea are summarized below:`

# Literature
## [N-BREF: A HIGH-FIDELITY DECOMPILER EXPLOITING PROGRAMMING STRUCTURES](https://ai.facebook.com/blog/introducing-n-bref-a-neural-based-decompiler-framework/)

current state of the art decompiler

paper: https://openreview.net/pdf?id=6GkL6qM3LV

code: https://github.com/facebookresearch/nbref?fbclid=IwAR3i37sj0z0YpLErZRoexZS_g2Oef-SspCwCxdPDdp-QQaYf1d2BOA0rtwY

## [Debin: predicting debug information](https://files.sri.inf.ethz.ch/website/papers/ccs18-debin.pdf)

Decompilation approaches based on neural machine translation (NMT)mechanism

## [Java decompiler diversity and its application to meta-decompilation](https://www.sciencedirect.com/science/article/abs/pii/S0164121220301151)

A survey on java decompilers, helps us identify their shortcomings

paper: https://reader.elsevier.com/reader/sd/pii/S0164121220301151?token=EDE1520D312B1F13F2F86C6E340D9F21BE11331D6D3F2662F75CD95A3BED26C897B0368BCC26BA3CE3BA60C73C635E1A&originRegion=eu-west-1&originCreation=20210920124041

## [Neutron: an attention-based neural decompiler](https://link.springer.com/content/pdf/10.1186/s42400-021-00070-0.pdf)

## [Improving type information inferred by decompilers with supervised machine learning](https://arxiv.org/pdf/2101.08116.pdf)

## [A Neural-based Program Decompiler](https://arxiv.org/pdf/1906.12029.pdf)

## [Evolving Exact Decompilation](http://storm-country.com/blog/evo-deco)

paper: https://www.cs.unm.edu/~eschulte/data/bed.pdf

## [Using Recurrent Neural Networks for Decompilation](https://www.cs.unm.edu/~eschulte/data/katz-saner-2018-preprint.pdf)

This article proposes an approach to completely reverse engineer binary blobs to C through RNN.

## [function reverse engineering with CNN](https://towardsdatascience.com/cnn-for-reverse-engineering-an-approach-for-function-identification-1c6af88bca43)

This article proposes an approach to identify functions from binary blobs through a CNN.

## [Talk on reverse engineering malware binary with ML](https://www.blackhat.com/docs/us-15/materials/us-15-Davis-Deep-Learning-On-Disassembly.pdf)

## https://github.com/nforest/awesome-decompilation

Useful resources on decompilation

## [Recognizing Functions in Binaries With Neural Networks](https://people.eecs.berkeley.edu/~dawnsong/papers/Recognizing%20functions%20in%20binaries%20with%20neural%20networks_augsut%202015.pdf)

## [Decompiling Obfuscated Android Applications](https://www.guardsquare.com/blog/decompiling-obfuscated-android-applications)

## [Kerberoid: A Practical Android Decompiler](https://seclab.skku.edu/wp-content/uploads/2019/11/p2557-jang.pdf)

## [Neural Reverse Engineering of Stripped Binaries using Augmented Control Flow Graphs](https://arxiv.org/pdf/1902.09122.pdf)

## [Towards Neural Decompilation](https://arxiv.org/pdf/1905.08325.pdf)

## [Machine Learning-Assisted Binary Code Analysis](http://pages.cs.wisc.edu/~jerryzhu/pub/nips07-abs.pdf)

## Probabilistic Disassembly
https://www.cs.purdue.edu/homes/zhan3299/res/ICSE19.pdf



(http://pages.cs.wisc.edu/~jerryzhu/pub/nips07-abs.pdf)