# CUDA_VISIBLE_DEVICES=3 python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2_from_scratch_l1+sobel --reset --relu
# CUDA_VISIBLE_DEVICES=4 python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_num8_featnum64_x2_relu_from_scratch --reset --relu


#  CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 3 --patch_size 144 --save edsr_baseline_x3_from_x2_mtlu --reset --pre_train ../experiment/edsr_baseline_x2_from_scratch_mtlu/model/model_best.pt

#  CUDA_VISIBLE_DEVICES=1 python main.py --model EDSR --scale 4 --patch_size 192 --save edsr_baseline_x4_from_x2_act_sa --reset --pre_train ../experiment/edsr_baseline_x2_from_scratch_act_sa/model/model_best.pt
# CUDA_VISIBLE_DEVICES=4 python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 2 --pre_train ../experiment/edsr_baseline_x2_from_scratch_xunit_new_6/model/model_best.pt --test_only
# CUDA_VISIBLE_DEVICES=4 python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 3 --pre_train ../experiment/edsr_baseline_x3_from_x2_xunit_new_6/model/model_best.pt --test_only
# CUDA_VISIBLE_DEVICES=4 python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train ../experiment/edsr_baseline_x4_from_x2_xunit_new_6/model/model_best.pt --test_only  

CUDA_VISIBLE_DEVICES=5 python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_num4_featnum32_x2_mnp_k8_from_scratch --reset
CUDA_VISIBLE_DEVICES=4 python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_num4_featnum32_x2_relu_from_scratch --reset --relu


CUDA_VISIBLE_DEVICES=0 python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 2 --pre_train ../experiment/edsr_baseline_num8_featnum64_x2_relu_from_scratch/model/model_best.pt --test_only --relu

CUDA_VISIBLE_DEVICES=0 python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2_from_scratch_xunit_same_parameter --reset