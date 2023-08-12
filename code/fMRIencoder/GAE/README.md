# GAE

## Run code
```
python train.py
```
* 위 명령어를 통해 코드를 실행할 수 있다.
* 코드 실행 전 data_dir의 경로를 데이터가 있는 폴더로 변경해주어야 한다.


## Run code with hyperparameters
```
python train.py --num_iter 100 --gnn_type GraphConv --masking True --masking_ratio 0.3 --hidden_layer 512 128
```
* 하이퍼파라미터를 설정하여 코드를 실행할 수 있다.
* 설정 가능한 하이퍼파라미터 목록과 각 파라미터의 디폴트 값은 opt.py 파일에서 확인 가능하다.