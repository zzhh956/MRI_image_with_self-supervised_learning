# MRI_image_with_self-supervised_learning

## Folder structure
    ```
    .
    ├── scripts
    │   ├── _pycache_/
    │   ├── dataloader.py
    │   ├── loss.py
    │   ├── main.py
    │   ├── models.py
    │   ├── parse_argue.py
    │   ├── test.py
    │   └── train.py
    ├── train_ckpt/
    ├── Readme.md
    ├── 311513015.npy
    └── requirements.txt
    ```

## Environment
- Python 3.6 or later version
    ```sh
    pip install -r requirements.txt
    ```

## Train
```sh
python ./scripts/main.py --train
```

## Make Prediction
Before you make a prediction, go to check ./train_ckpt folder and get the ith epoch you want.
```sh
python main.py --test --epoch <epoch>
```
The output file is `311513015.npy`.