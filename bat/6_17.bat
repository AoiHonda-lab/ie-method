@ L2_15_15 L3_5_5_5 L3_15_15_15

start  /min cmd /k python main.py --model ie --epoch 200 --data_model _3_3_pool_1_mnist_2class --lr 0.01 --opt adam --func sigmoid --day 6_17 --ie_op shoki --add 1
start  /min cmd /k python main.py --model ie --epoch 200 --data_model _3_3_pool_1_mnist_2class --lr 0.01 --opt adam --func sigmoid --day 6_17 --ie_op shoki --add 2
start  /min cmd /k python main.py --model ie --epoch 200 --data_model _3_3_pool_1_mnist_2class --lr 0.01 --opt adam --func sigmoid --day 6_17 --ie_op shoki --add 3
start  /min cmd /k python main.py --model ie --epoch 200 --data_model _3_3_pool_1_mnist_2class --lr 0.01 --opt adam --func sigmoid --day 6_17 --ie_op shoki --add 9
start  /min cmd /k python main.py --model ie --epoch 200 --data_model _3_3_pool_1_mnist_2class --lr 0.01 --opt adam --func relu --day 6_17 --ie_op shoki --add 2
start  /min cmd /k python main.py --model mlp --epoch 200 --data_model _3_3_pool_1_mnist_2class --lr 0.01 --opt adam --func sigmoid --day 6_17 --ie_op shoki --add 2

start  /min cmd /k python main_lbl.py --model mlp --epoch 200 --data_model _3_3_pool_1_mnist_2class --lr 0.01 --opt adam --func sigmoid --day 6_17 --ie_op shoki --add 9
start  /min cmd /k python main_lbl.py --model mlp --epoch 200 --data_model _3_3_pool_1_mnist_2class --lr 0.001 --opt adam --func sigmoid --day 6_17 --ie_op shoki --add 9
start  /min cmd /k python main_lbl.py --model mlp --epoch 200 --data_model _3_3_pool_1_mnist_2class --lr 0.01 --opt sgd --func sigmoid --day 6_17 --ie_op shoki --add 9
start  /min cmd /k python main_lbl.py --model mlp --epoch 200 --data_model _3_3_pool_1_mnist_2class --lr 0.001 --opt adam --func sigmoid --day 6_17 --ie_op shoki --add 9
start  /min cmd /k python main_lbl.py --model mlp --epoch 200 --data_model _3_3_pool_1_mnist_2class --lr 0.01 --opt adam --func relu --day 6_17 --ie_op shoki --add 9
start  /min cmd /k python main_lbl.py --model mlp --epoch 200 --data_model _3_3_pool_1_mnist_2class --lr 0.001 --opt adam --func relu --day 6_17 --ie_op shoki --add 9
start  /min cmd /k python main_lbl.py --model mlp --epoch 200 --data_model _3_3_pool_1_mnist_2class --lr 0.01 --opt sgd --func relu --day 6_17 --ie_op shoki --add 9
start  /min cmd /k python main_lbl.py --model mlp --epoch 200 --data_model _3_3_pool_1_mnist_2class --lr 0.001 --opt adam --func relu --day 6_17 --ie_op shoki --add 9