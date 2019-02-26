# Supervised-Morphological-Segmentation
Python implementation of *Supervised Morphological Segmentation in a Low-Resource Learning Setting using Conditional Random Fields (Ruokolainen Teemu, et al. 2013)* 

First homework of the Natural Language Processing course, prof. Roberto Navigli.

University project • 2016 - Natural Language Processing - MSc in Computer Science, I year 

The source code is is accompanied by a report of the performed experiments (`report.pdf`), the original article can be found [here](http://www.aclweb.org/anthology/W13-3504). 

## Article abstract

We discuss data-driven morphological
segmentation, in which word forms are
segmented into morphs, the surface forms
of morphemes. Our focus is on a lowresource learning setting, in which only a
small amount of annotated word forms are
available for model training, while unannotated word forms are available in abundance. The current state-of-art methods 1) exploit both the annotated and unannotated data in a semi-supervised manner, and 2) learn morph lexicons and subsequently uncover segmentations by generating the most likely morph sequences.
In contrast, we discuss 1) employing only
the annotated data in a supervised manner, while entirely ignoring the unannotated data, and 2) directly learning to predict morph boundaries given their local
sub-string contexts instead of learning the
morph lexicons. Specifically, we employ conditional random fields, a popular
discriminative log-linear model for segmentation. We present experiments on
two data sets comprising five diverse languages. We show that the fully supervised boundary prediction approach outperforms the state-of-art semi-supervised
morph lexicon approaches on all languages when using the same annotated
data sets.

## Implementation details

The language chosen is Python 2.7, the library used are the sklearn_crfsuite library to implement the CRF and
the pickle library to export the model.
Following the model, the classification problem is well defined when each character of the dataset is
represented by a binary vector of features and labeled with the appropriate class among the possible six. The
feature vector is encoded as a dictionary, theoretically it should be very sparse with few 1 and a large amount
of 0, nevertheless the crf_suite library is able to receive as input only the present features avoiding to build and
deal with huge dictionaries.
The dictionaries of each character of a word are organized in a list and all these lists are organized in another
list representing the desired learning set (training, dev or test). The same procedure is performed for the labels,
with a string containing the label instead of the dictionary. The described data structure coincides with the
accepted input format of the CRF class of the crf_suite library.
Referring to the main.py file, the acquisition of the datasets from the given files is performed in the first section
of the code `#COLLECT DATA AND LABELLING`. The construction of the feature dictionaries and the
organization in the correct data structure is accomplished by the prepare_data function defined in the
`#COMPUTE FEATURES` section and used in the `#DATA PREPARATION AND FIT` section. In the final
`#EVALUATION` section the Precision, Recall and F1 scores are computed and the results are printed on the
console.

## Results
Results are in general satisfactory, with a F1 score near 0.80. The results obtained with the extra features are slightly
better, especially with few samples where improvement
on the F1 score is up to a 3%.

See the report for a detailed discussion of the results and to see scores and graphs.


