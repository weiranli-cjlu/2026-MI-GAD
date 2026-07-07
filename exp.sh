python run.py --dataset BlogCatalog \
    --hidden_dim 256 --epoch 100 \
    --lr 0.0011375443848145328 --dropout 0.4201491411392214 \
    --alpha 0.5045263236715447 --beta 0.9991322785603456 \
    --weight_decay 0.00045615156408623656 \
    --runs 10 --result-csv outputs/result.csv --tqdm --use_best

python run.py --dataset book \
    --hidden_dim 128 --epoch 100 \
    --lr 0.006151185471717628 --dropout 0.20197536131898014 \
    --alpha 1.0077688812140018 --beta 0.7858069872119468 \
    --weight_decay 4.811676811374437e-05 \
    --runs 10 --result-csv outputs/result.csv --tqdm --use_best

python run.py --dataset cora \
    --hidden_dim 256 --epoch 100 \
    --lr 0.0009718319944817398 --dropout 0.3136396976291964 \
    --alpha 1.1413115275378245 --beta 0.07414817040689042 \
    --weight_decay 3.4630370261191806e-08 \
    --runs 10 --result-csv outputs/result.csv --tqdm --use_best

python run.py --dataset Flickr \
    --hidden_dim 256 --epoch 100 \
    --lr 0.00010600355282680341 --dropout 0.29240716028243685 \
    --alpha 1.4056677888641884 --beta 0.05138493873450106 \
    --weight_decay 6.814148625663892e-08 \
    --runs 10 --result-csv outputs/result.csv --tqdm --use_best

python run.py --dataset tolokers \
    --hidden_dim 256 --epoch 100 \
    --lr 0.00778211923530846 --dropout 0.009726619495385256 \
    --alpha 1.2859503583425633 --beta 0.7183848413517264 \
    --weight_decay 0.0005631694744223225 \
    --runs 10 --result-csv outputs/result.csv --tqdm --use_best

python run.py --dataset twitter \
    --hidden_dim 64 --epoch 100 \
    --lr 0.002514036246129349 --dropout 0.22082722488581769 \
    --alpha 1.09383722174718 --beta 0.20121090459761076 \
    --weight_decay 6.922076695703508e-07 \
    --runs 10 --result-csv outputs/result.csv --tqdm --use_best

python run.py --dataset YelpChi \
    --hidden_dim 256 --epoch 100 \
    --lr 0.006992264840139406 --dropout 0.010794512483552843 \
    --alpha 1.2820739072662677 --beta 0.10848262963252014 \
    --weight_decay 0.0002459575018195691 \
    --runs 10 --result-csv outputs/result.csv --tqdm --use_best
