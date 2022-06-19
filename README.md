# ReSel

This repo contains the code for paper "ReSel: Learning to Retrieve and Select for Multi-Modal Scientific Information Extraction"

---

## Dependency

This repo is built with Python 3.8.8. Please check `./requirements.txt` for required Python packages. Other packages versions are not tested.

## Datasets

We use data provided by [SciREX](https://github.com/allenai/SciREX), [PubMed](https://hanover.azurewebsites.net/downloads/naacl2019.aspx), and [TDMS](https://github.com/IBM/science-result-extractor). As SciREX and PubMed are pure text datasets, we also provide the corresponding table extractor from raw document files for both of them. You can find them in `./datasets/<DATASET_NAME>_table_extractor.py`.

We will upload the pre-processed datasets on the cloud drive later, as they exceed the storage limitation of the Github repo. You can either download the original data from the above links, or download our pre-processed version once we make it public. The downloaded datasets should be placed in `./datasets/<DATASET_NAME>/`.

## Run

### High-Level Retriever Experiments

Key Arguments:

* `lr`: The learning rate of the model
* `saved_embed`: Whether to read the previously saved and pre-processed data
* `shuffle`: Whether to re-shuffle the dataset or not
* `duplicate`: Whether to allow multiple queries for one document or not
* `epochs`: number of the epochs
* `seed`: Random seed
* `saved_embed2`: Whether to use some pre-computed values or not. Once you have computed once, you are able to find them in `./saved_model/saved_embed/`

```python
python high_level_ReSel.py [ARGUMENTS]
```

Example:

```python
CUDA_VISIBLE_DEVICES=0 python high_level_ReSel.py --lr 1e-4 --saved_embed y --embed_style paragraph --partial --query_style question --table_style caption+table --shuffle --duplicate y --softedges --epochs 50 --seed 1 --saved_embed2 n
```

### Low-Level Entity Extractor Experiments

```python
python low_level_main.py [ARGUMENTS]
```

Example:

```python
CUDA_VISIBLE_DEVICES=0 python low_level_main.py --model ReSel --lr 1e-3 --saved_embed y --embed_style paragraph --partial --query_style question --table_style caption+table --shuffle --duplicate y --softedges --epochs 50 --edges ccr --evaluation 5 --seed 1
```

### Overall Performance

Key Arguments:

* `llr`: The learning rate of the low-level model
* `lepochs`: The number of epochs for low-level model

```python
python overall_main.py [ARGUMENTS]
```

Example:

```python
CUDA_VISIBLE_DEVICES=0 python overall_main.py --model ReSel --llr 1e-3 --saved_embed y --embed_style paragraph --partial --query_style question --table_style caption+table --shuffle --duplicate y --softedges --epochs 50 --edges ccr --evaluation 5 --seed 1 --lepochs 50 --saved_embed2 y
```