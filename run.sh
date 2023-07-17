# LPN  
# python train.py \
# --name='three_view_long_share_d0.75_256_s1_google_LPN4_lr0.001' \
# --data_dir='/home/wangtyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --fp16 \
# --LPN \
# --extra \
# --block=4 \
# --lr=0.001 \
# --gpu_ids='0'

# python test.py \
# --name='three_view_long_share_d0.75_256_s1_google_LPN4_lr0.001' \
# --test_dir='/home/wangtyu/datasets/University-Release/test' \
# --batchsize=128 \
# --gpu_ids='0'

# Baseline
python train.py \
--name='three_view_long_share_d0.75_256_s1_google_lr0.01' \
--data_dir='/home/wangtyu/datasets/University-Release/train' \
--views=3 \
--droprate=0.75 \
--extra \
--share \
--stride=1 \
--h=256 \
--w=256 \
--fp16 \
--lr=0.01 \
--gpu_ids='0'

python test.py \
--name='three_view_long_share_d0.75_256_s1_google_lr0.01' \
--test_dir='/home/wangtyu/datasets/University-Release/test' \
--gpu_ids='1'

# two view(drone+satellite) + LPN + gem_pool
# python train.py \
# --name='two_view_long_no_street_share_d0.75_256_s1_LPN4_lr0.001_gem' \
# --data_dir='/home/wangtyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --fp16 \
# --LPN \
# --pool='gem' \
# --block=4 \
# --lr=0.001 \
# --gpu_ids='1'

# python test.py \
# --name='two_view_long_no_street_share_d0.75_256_s1_LPN4_lr0.001_gem' \
# --test_dir='/home/wangtyu/datasets/University-Release/test' \
# --batchsize=128 \
# --gpu_ids='1'

# baseline + netvlad(pooling)
# python train.py \
# --name='three_view_long_share_d0.75_256_s1_google_lr0.01_netvlad' \
# --data_dir='/home/wangtyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --fp16 \
# --lr=0.01 \
# --pool='netvlad' \
# --gpu_ids='1'

# python test.py \
# --name='three_view_long_share_d0.75_256_s1_google_lr0.01_netvlad' \
# --test_dir='/home/wangtyu/datasets/University-Release/test' \
# --batchsize=64 \
# --gpu_ids='1'