@ L2_15_15 L3_5_5_5 L3_15_15_15

start  /min cmd /k python main_ver2.py --model ie --out 1 --lossf ent --batch_size 128 --epoch 5000 --data_model _3_3_pool_1_mnist_2class --lr 0.01 --opt adam --func sigmoid --day 7_25_mono --add 2 --not_trans_flag 
start  /min cmd /k python main_ver2.py --model ie --out 1 --lossf mse --batch_size 128 --epoch 5000 --data_model _3_3_pool_1_mnist_2class --lr 0.01 --opt adam --func sigmoid --day 7_25_mono --add 2 --not_trans_flag
start  /min cmd /k python main_ver2.py --model ie --out 1 --lossf mse_sig --batch_size 128 --epoch 5000 --data_model _3_3_pool_1_mnist_2class --lr 0.01 --opt adam --func sigmoid --day 7_25_mono --add 2 --not_trans_flag