Pretrain命令：
CUDA_VISIBLE_DEVICES=9 python pretrain_moco_mask.py --lr 0.02 --batch-size 128 --teacher-t 0.05 --student-t 0.1 --topk 16384 --mlp --contrast-t 0.07 --contrast-k 16384 --checkpoint-path /mnt/netdisk/zhangjh/Code/CMD-main/mask_checkpoints/both_prompt_r08 --schedule 351 --epochs 451 --pre-dataset ntu60 --skeleton-representation graph-based --protocol cross_view --exp-descri Mask_reconstruction_both_prompt_small_wsgencoder

使用JointMask3 ratio = 0.6
两个prompt，task-prompt的维度为128*2 = 256

motion和bone没有使用prompt

finetune命令除了Retrieval和cmd保持一致，其余的都是在HiCLR_DDP中，还有detection的实验在skeleton_detection中。

weights:
xview:
/mnt/netdisk/zhangjh/Code/CMD-main/mask_checkpoints/both_prompt_128*2_sgenc/

/mnt/netdisk/zhangjh/Code/CMD-main/mask_checkpoints/pretrain_mask_wcl/bone/JM3_ratio06_w40/checkpoint_0450.pth.tar

/mnt/netdisk/zhangjh/Code/CMD-main/mask_checkpoints/pretrain_mask_wcl/motion/JM3_w40

xsub:
/mnt/netdisk/zhangjh/Code/CMD-main/mask_checkpoints/both_prompt_128*2_sgenc
/mnt/netdisk/zhangjh/Code/CMD-main/mask_checkpoints/xsub/pretrain_mask_wcl/bone/JM3_W40_R06
/mnt/netdisk/zhangjh/Code/CMD-main/mask_checkpoints/xsub/pretrain_mask_wcl/motion/JM3_w40