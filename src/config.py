def get_default_cfg():
    return {
        # model & setting
        "arch": "dinov3",     # ["dinov3", "resnet18", "resnet152", "yololiked"]
        "mode": "zeroshot",     # ["zeroshot", "fewshot"]
        "fullsup": False,
        "all_regions": ["gaziantep","hatay","kahramanrarus","kirikhan","nurdagi","sackcagozu","satirhuyuk","sekeroba","turkoglu"],
        "source_regions": ["hatay","kahramanrarus","kirikhan","nurdagi","sackcagozu","satirhuyuk","sekeroba","turkoglu"],
        "target_regions": ["gaziantep"],

        # param
        "train_pct": 1,
        "test_prob_thr": 0.5,
        "img_size": 128,
        "batch_size": 32,
        "num_workers": 8,
        "epochs": 10,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,

        # kfold
        "kfold_k": 5,
        "kfold_seed": 42,
        "kfold_stratify": True,
        "kfold_fold_start": 0,
        "kfold_fold_end": 5,
        
        # loss
        "focal_alpha": 0.70,
        "focal_gamma": 2.0,
        "lambda_pc": 0,
        "lambda_dpt": 0,

        # pc
        "k_pos_pixel": 32,
        "k_neg_pixel": 32,
        "max_pixels_per_image": 1024,
        "sample_pos_ratio": 0.5,
        "pixel_batch_size": 16,
        "pixel_num_workers": 8,
        "pixel_normalize": True,

        # dpt
        "patch_grid": 4,
        "dpt_margin": 0.2,
        "dpt_triplets_per_img": 16,
        "dpt_use_nonbuilding_as_neg": True,
        "normalize": True,
        "all_touched": True,

        # fewshot
        "fewshot_n": 1,
        "fewshot_lr": 3e-5,
        "fewshot_epochs": 3,
        "fewshot_seed": 0,
        "lora_r": 8,
        "lora_alpha": 16.0,
        "lora_dropout": 0.0,

        # path
        "save_dir": "/mnt/liwei/turkey/DINOV3/checkpoints/turkey/fullsup-new/dinov3-128-ssdcpc",
        "data_root": "./data/smart_transfer_data_128",
        "footprint_dir": "./data/GlobalBuildingAtlas",
        "geo_json_dir": "./data/tiles_meta",
        "repo_dir": "./backbone/dinov3",
        "weight_path": "./backbone/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    }