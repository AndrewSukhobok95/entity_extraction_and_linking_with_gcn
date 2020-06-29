# DL_GraphEntity_project

Research Experience Project 02/2020 - 05/2020

Aim: Train DL model to extract facts from text.

Done:
- DL model is built and tested (NER - good quality, RE - relatively bad)
- classes for metrics calculation are written (model/metrics.py)

ToDo:
- finish testing the model, calculate metrics
- clean the dataset
- possible bug with model saving and loading (!)

Example:

> ORIGINAL SENTENCE:
>
> "In remote rural corners of India , particularly in conflict zones like Chhattisgarh , police ranks are woefully understaffed , and isolated police posts are among the rebels ' favorite targets ."
>
> TRUE ENTITIES:
> + LOCATION - India
> + LOCATION - Chhattisgarh
>
> PREDICTED ENTITIES:
> + B-LOCATION - india
> + B-LOCATION - chhattisgarh
>
> SENTENCE WITH PREDICTED ENTITIES:
>
> "in remote rural corners of india (B-LOCATION) , particularly in conflict zones like chhattisgarh (B-LOCATION) , police ranks are woefully understaffed , and isolated police posts are among the rebels ' favorite targets . "
>
> TRUE RELATIONS:
> + India - Chhattisgarh - /location/location/contains
> + Chhattisgarh - India - None
>
> PREDICTED RELATIONS:
> + india - chhattisgarh - /location/location/contains



Architecture in this project is inspired by:
* Original paper: https://www.aclweb.org/anthology/P19-1136/
* Original project page: https://tsujuifu.github.io/projs/acl19_graph-rel.html

The following dataset was used: https://github.com/INK-USC/USC-DS-RelationExtraction

Training parameters:

| Parameters    | Values        |
| ------------- |:-------------:|
| N epochs      | 100           |

---
#### Useful links:

* BERT link:
    * https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

---
#### Related papers:

* Span BERT
    * https://arxiv.org/abs/1909.07755
    * https://github.com/markus-eberts/spert/tree/master/spert

* GUpdater
    * https://www.aclweb.org/anthology/D19-1265.pdf
    * https://github.com/esddse/GUpdater

* DynamicSpanGraphs
    * https://arxiv.org/pdf/1904.03296v1.pdf
    * https://github.com/luanyi/DyGIE

* Inter Sentence RE GCN
    * https://www.aclweb.org/anthology/P19-1423/

* A Hierarchical Framework for Relation Extraction with Reinforcement Learning
    * https://arxiv.org/abs/1811.03925
    * https://github.com/truthless11/HRL-RE

---
#### Datasets:

Ready datastes:

* Preprocessed dataset - NYT, PubMed-BioInfer, Wiki-KBP (json): 

	* https://github.com/INK-USC/USC-DS-RelationExtraction

* NYT and Google IIS (different preprocessing - described inside):

	* https://github.com/malllabiisc/RESIDE

* Several public datasets (json):

	* https://github.com/dstl/re3d

* Synthetic data:

    * http://knowitall.cs.washington.edu/leibniz/

Other Datasets and Info:

* Aggeragation page for many datsets: https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets/blob/master/README.md

* NYT (unprocessed): http://iesl.cs.umass.edu/riedel/ecml/

* Google QA dataset: https://github.com/google-research-datasets/wiki-reading

* Open IE page: https://openie.allenai.org/

* Max-Planc Institute info: https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/

NER datasets:

* https://github.com/juand-r/entity-recognition-datasets#id5

RE related datasets:

* https://github.com/sahitya0000/Relation-Classification
