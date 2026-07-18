from __future__ import annotations

from pathlib import Path

from .provider_contract import EMBEDDING_DIMENSION


class SoccerNetInferenceDataset:
    """The minimal SoccerNet dataset metadata required for PRTReID inference."""

    masks_dirs = {
        "gaussian_joints": (10, False, ".npy", [f"p{part}" for part in range(1, 17)]),
        "gaussian_keypoints": (17, False, ".npy", [f"p{part}" for part in range(1, 17)]),
        "pose_on_img": (35, False, ".npy", [f"p{part}" for part in range(1, 35)]),
    }

    @classmethod
    def get_masks_config(cls, masks_dir: str):
        return cls.masks_dirs.get(masks_dir)


def register_soccernet_inference_dataset(prtreid_module) -> None:
    """Idempotently expose SoccerNet metadata to PRTReID's registry."""

    from prtreid.data.datasets import get_image_dataset

    try:
        get_image_dataset("SoccerNet")
    except ValueError:
        prtreid_module.data.register_image_dataset(
            "SoccerNet", SoccerNetInferenceDataset, "sn"
        )


def reference_config(weights_path: Path, model_root: Path, use_gpu: bool) -> dict:
    """Concrete inference config matching the SoccerNet PRTReID baseline."""

    return {
        "use_gpu": use_gpu,
        "project": {
            "name": "ReplayStudioIdentityWorker",
            "experiment_name": "inference",
            "notes": "",
            "tags": [],
            "job_id": "0",
            "logger": {"use_tensorboard": False, "use_wandb": False},
        },
        "data": {
            "root": str(model_root),
            "type": "image",
            "sources": ["SoccerNet"],
            "targets": ["SoccerNet"],
            "height": 256,
            "width": 128,
            "combineall": False,
            "transforms": ["rc", "re"],
            "save_dir": "/tmp/prtreid",
            "workers": 0,
        },
        "sampler": {
            "train_sampler": "PrtreidSampler",
            "train_sampler_t": "PrtreidSampler",
            "num_instances": 4,
        },
        "model": {
            "name": "bpbreid",
            "pretrained": True,
            "save_model_flag": False,
            "load_config": True,
            "load_weights": str(weights_path),
            "bpbreid": {
                "pooling": "gwap",
                "normalization": "identity",
                "mask_filtering_training": False,
                "mask_filtering_testing": False,
                "training_binary_visibility_score": True,
                "testing_binary_visibility_score": True,
                "last_stride": 1,
                "learnable_attention_enabled": False,
                "dim_reduce": "after_pooling",
                "dim_reduce_output": EMBEDDING_DIMENSION,
                "backbone": "hrnet32",
                "test_embeddings": ["globl"],
                "test_use_target_segmentation": "none",
                "shared_parts_id_classifier": False,
                "hrnet_pretrained_path": str(model_root),
                "masks": {"type": "disk", "dir": "", "preprocess": "id"},
            },
        },
        "loss": {
            "name": "part_based",
            "part_based": {
                "name": "part_averaged_triplet_loss",
                "ppl": "cl",
                "weights": {
                    "globl": {"id": 1.0, "tr": 1.0},
                    "foreg": {"id": 0.0, "tr": 0.0},
                    "conct": {"id": 0.0, "tr": 0.0},
                    "parts": {"id": 0.0, "tr": 0.0},
                    "pixls": {"ce": 0.0},
                },
            },
        },
        "train": {"batch_size": 32, "max_epoch": 20},
        "test": {
            "evaluate": True,
            "detailed_ranking": False,
            "start_eval": 40,
            "batch_size": 64,
            "batch_size_pairwise_dist_matrix": 5000,
            "normalize_feature": True,
            "dist_metric": "euclidean",
            "visrank": False,
            "visrank_per_body_part": False,
            "vis_embedding_projection": False,
            "vis_feature_maps": False,
            "visrank_topk": 10,
            "visrank_count": 0,
            "visrank_q_idx_list": [],
            "part_based": {"dist_combine_strat": "mean"},
        },
    }
