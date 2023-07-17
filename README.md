## MLPN
### Our solution for the ACMMM23 Multimedia Drone Satellite Matching Challenge
### Train
python train_info.py \
--name='name' \
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
--batchsize=8 \
--SAM=1 \
--infonce=1 \
--DA \
--color_jitter

### Test
python test_160k.py \
--name='name' \
--batchsize=256 \
--gpu_ids='1'
