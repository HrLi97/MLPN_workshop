## baseline + DWDR ##
# python train_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.01_decouple_balance_lambda768_e11' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.01 \
# --balance \
# --decouple \
# --fp16 \
# --e1=1 \
# --e2=1 \
# --g=0.9 \
# --lambd=0.0013 \
# --experiment_name='two_view_long_share_d0.75_256_s1_lr0.01_decouple_balance_lambda768_e11' \
# --gpu_ids='1' \

# python test_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.01_decouple_balance_lambda768_e11' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --gpu_ids='1'

## baseline + Barlow Twins ##
# python train_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.01_decouple_balance_lambda768_e00' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=2 \
# --droprate=0.75 \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.01 \
# --balance \
# --decouple \
# --fp16 \
# --e1=0 \
# --e2=0 \
# --g=0.9 \
# --lambd=0.0013 \
# --experiment_name='two_view_long_share_d0.75_256_s1_lr0.01_decouple_balance_lambda768_e00' \
# --gpu_ids='0' \

# python test_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.01_decouple_balance_lambda768_e00' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --gpu_ids='0'

## LPN + DWDR ##
# python train_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.003_lpn4_balance_decouple_e11_lambd768' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --LPN \
# --block=4 \
# --lr=0.003 \
# --fp16 \
# --experiment_name='two_view_long_share_d0.75_256_s1_lr0.003_lpn4_balance_decouple_e11_lambd768' \
# --gpu_ids='1' \
# --g=0.9 \
# --lambd=0.0013 \
# --balance \
# --decouple \
# --e1=1 \
# --e2=1 \

# python test_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.003_lpn4_balance_decouple_e11_lambd768' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --gpu_ids='1'

swin-b + DWDR ##
python train_mul_gpu.py \
--name='swin-b+dwdr_false_200' \
--data_dir='/home/lihaoran/BJDD_datesets/datesets/University-Release/train' \
--views=3 \
--droprate=0.75 \
--share \
--stride=1 \
--h=224 \
--w=224 \
--lr=0.005 \
--balance \
--decouple \
--e1=1 \
--e2=1 \
--g=0.9 \
--lambd=0.002 \
--experiment_name='swin-b+dwdr_false_200' \
--gpu_ids='1' \
--swin 

python test_mul_gpu.py \
--name='swin-b+dwdr_false' \
--test_dir='/home/lihaoran/BJDD_datesets/datesets/University-Release/test' \
--gpu_ids='3'

# Baseline expand ID
# python train_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_google_lr0.01_expand-3id_batch16' \
# --data_dir='/home/lvbinbin/wty/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.01 \
# --experiment_name='two_view_long_share_d0.75_256_s1_google_lr0.01_expand-3id_batch16' \
# --gpu_ids='3' \
# --fp16 \
# --expand_id \
# --seed=3 \
# --batchsize=16 

# python test_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_google_lr0.01_expand-3id_batch16' \
# --test_dir='/home/lvbinbin/wty/datasets/University-Release/test' \
# --gpu_ids='3'

# Baseline expand batchsize 
# python train_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_google_lr0.01_batchsize16' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --batchsize=16 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.01 \
# --experiment_name='two_view_long_share_d0.75_256_s1_google_lr0.01_batchsize16' \
# --gpu_ids='1' \
# --fp16 \
# --normal \
# --seed=3

# python test_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_google_lr0.01_batchsize16' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --gpu_ids='1'



## Baseline + satellite based sampling ##
# python train_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.01_batch8_sat_leading' \
# --data_dir='/home/lvbinbin/wty/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.01 \
# --balance \
# --e1=1 \
# --e2=1 \
# --g=0.9 \
# --fp16 \
# --lambd=0.0013 \
# --experiment_name='two_view_long_share_d0.75_256_s1_lr0.01_batch8_sat_leading' \
# --gpu_ids='3' \
# --sat_lead \
# --seed=3 \
# --batchsize=8

# python test_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.01_batch8_sat_leading' \
# --test_dir='/home/lvbinbin/wty/datasets/University-Release/test' \
# --gpu_ids='3'

# train ours  cswin=2 swinv2=1
python train_mul_gpu.py \
--name='swinv2_test_e200' \
--data_dir='/home/lihaoran/datasets/University-Release/train/' \
--views=3 \
--droprate=0.75 \
--share \
--stride=1 \
--h=224 \
--w=224 \
--lr=0.005 \
--e1=1 \
--e2=1 \
--g=0.9 \
--lambd=0.0013 \
--experiment_name='cswin_5.30_e200' \
--gpu_ids='1' \
--seed=3 \
--modelNum=1


#workshop
python test_160k.py \
--name='lhr_6.15_swinv2_layer2_sam_test0_203' \
--batchsize=256 \
--gpu_ids='1'

#256 256
python train_mul_gpu.py \
--name='cswin_5.30_e200' \
--data_dir='/home/lihaoran/datasets/University-Release/train/' \
--views=3 \
--droprate=0.75 \
--share \
--stride=1 \
--h=256 \
--w=256 \
--lr=0.005 \
--e1=1 \
--e2=1 \
--g=0.9 \
--lambd=0.0013 \
--experiment_name='swinV2_5.30' \
--gpu_ids='2' \
--seed=3


train swin
python train_mul_gpu.py \
--name='mulGPU_6.12_balance_test1' \
--data_dir='/home/lihaoran/datasets/University-Release/train/' \
--views=3 \
--droprate=0.75 \
--share \
--stride=1 \
--h=224 \
--w=224 \
--lr=0.005 \
--e1=1 \
--e2=1 \
--g=0.9 \
--lambd=0.0013 \
--experiment_name='swin-test-5.25' \
--gpu_ids='3' \
--swin \
--seed=3 \
--LPN \
--balance

CUDA_VISIBLE_DEVICES=2 python test_mul_gpu.py \
--name='lhr_6.27_threeIn_61218_*54_203' \
--test_dir='/home/lihaoran/BJDD_datesets/datesets/University-Release/test' \
--gpu_ids='2' \
--batchsize=128


CUDA_VISIBLE_DEVICES=3 python train_std.py \
--name='lhr_7.12_balance_test1_203' \
--data_dir='/home/lihaoran/BJDD_datesets/datesets/University-Release/train/' \
--views=3 \
--droprate=0.75 \
--share \
--stride=1 \
--h=256 \
--w=256 \
--lr=0.005 \
--e1=1 \
--e2=1 \
--g=0.9 \
--lambd=0.0013 \
--experiment_name='swin-test-5.25' \
--gpu_ids='3' \
--swin \
--seed=3 \
--LPN \
--balance \
--batchsize=8

python train_samlhr.py \
--name='samlhr_6.13_test1' \
--data_dir='/home/lihaoran/BJDD_datesets/datesets/University-Release/train/' \
--views=3 \
--droprate=0.75 \
--share \
--stride=1 \
--h=224 \
--w=224 \
--lr=0.005 \
--e1=1 \
--e2=1 \
--g=0.9 \
--lambd=0.0013 \
--experiment_name='swin-test-5.25' \
--gpu_ids='2' \
--swin \
--seed=3 \
--LPN \
--SAM=0 \
--infonce=0 \


--balance \


--decouple


--balance \


# python test_mul_gpu.py \
# --name='three_view_swin-b_long_share_d0.75_224_s1_lr0.005' \
# --test_dir='/home/lihaoran/BJDD_datesets/datesets/University-Release/test' \
# --gpu_ids='1'



# train half channel dwdr
# python train_mul_gpu_half.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.01_balance_decouple_lambda256_e11_g0.9_0.5channel' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.01 \
# --balance \
# --decouple \
# --e1=1 \
# --e2=1 \
# --g=0.9 \
# --fp16 \
# --lambd=0.0039 \
# --experiment_name='two_view_long_share_d0.75_256_s1_lr0.01_balance_decouple_lambda256_e11_g0.9_0.5channel' \
# --gpu_ids='0' \
# --seed=3 

# python test_mul_gpu.py \
# --name='two_view_long_share_d0.75_256_s1_lr0.01_balance_decouple_lambda256_e11_g0.9_0.5channel' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --gpu_ids='0'
