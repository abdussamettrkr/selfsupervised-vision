## Selfsupervised Learning in Computer Vision


## Existing Architectures

 - (vit) [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
 - (Resnet) Resnet18-34-50

## Requirements

- Python (**>=3.8**)
- PyTorch (**>=1.11**)
- Other dependencies (pyyaml)

## Usage

Simply run the cmd for the training:

```bash
python train_.py --data-path /data/bird_dataset --epochs 100 --batch-size 48 --config-file ./models/model_name/model_name.yaml
```


We use yaml files to keep track of model details.


