"""Configuration surface for experiment presets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50_257
    max_seq_len: int = 512
    n_layers: int = 6
    n_heads: int = 4
    d_model: int = 128
    ffn_dim: int = 512
    dropout: float = 0.0
    rope_base: int = 10_000
    embedding_init_std: float = 0.02
    prediction_head_init_std: float = 0.02
    output_scale_init: float = 1.0


@dataclass(frozen=True)
class OptimConfig:
    learning_rate: float = 3e-4
    ce_embedding_lr_scale: float = 0.1
    l2_embedding_lr_scale: float = 1.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_frac: float = 0.05
    decay_frac: float = 0.2
    min_lr_ratio: float = 0.1
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "HuggingFaceFW/finewiki"
    dataset_split: str = "train"
    dataset_languages: tuple[str, ...] = (
        "en",
        "ar",
        "zh",
        "fr",
        "de",
        "ja",
        "ko",
        "es",
    )
    streaming: bool = True
    use_cache: bool = True
    cache_dir: str = "data/cache"
    batch_size: int = 16
    eval_batch_size: int = 16
    tokenizer_name: str = "lfm25"
    max_seq_len: int = 512
    train_token_budget: int = 262_144
    val_token_budget: int = 32_768
    max_train_documents: int | None = 4_096
    max_val_documents: int | None = 512
    num_workers: int = 0
    pin_memory: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"
    dtype: str = "float32"
    train_steps: int = 100
    eval_every: int = 20
    checkpoint_every: int = 50
    log_every: int = 1
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    assert_target_embedding_grads: bool = False
    log_gradient_metrics: bool = True


@dataclass(frozen=True)
class ObjectiveConfig:
    name: str
    lambda_sigreg: float = 0.0
    sigreg_warmup_steps: int = 0
    num_slices: int = 1_024
    epps_pulley_points: int = 17
    learned_output_scale: bool = False
    sigreg_random_vocab_size: int = 0
    sigreg_include_active_predictions: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    objective: ObjectiveConfig
    model: ModelConfig
    optim: OptimConfig
    data: DataConfig
    runtime: RuntimeConfig

    @property
    def output_path(self) -> Path:
        return Path(self.runtime.output_dir)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.runtime.checkpoint_dir) / self.name


def _base_experiment(
    name: str, objective_name: str, lambda_sigreg: float
) -> ExperimentConfig:
    model = ModelConfig()
    data = DataConfig(max_seq_len=model.max_seq_len)
    return ExperimentConfig(
        name=name,
        objective=ObjectiveConfig(name=objective_name, lambda_sigreg=lambda_sigreg),
        model=model,
        optim=OptimConfig(),
        data=data,
        runtime=RuntimeConfig(),
    )


def _l2_model_config() -> ModelConfig:
    return replace(
        ModelConfig(),
        embedding_init_std=0.01,
        prediction_head_init_std=0.0,
    )


def _l2_model_config_with_embedding_std(embedding_init_std: float) -> ModelConfig:
    return replace(
        ModelConfig(),
        embedding_init_std=embedding_init_std,
        prediction_head_init_std=0.0,
    )


def _with_runtime(
    config: ExperimentConfig, **runtime_updates: object
) -> ExperimentConfig:
    return replace(config, runtime=replace(config.runtime, **runtime_updates))


def _with_data(config: ExperimentConfig, **data_updates: object) -> ExperimentConfig:
    return replace(config, data=replace(config.data, **data_updates))


def _with_optim(config: ExperimentConfig, **optim_updates: object) -> ExperimentConfig:
    return replace(config, optim=replace(config.optim, **optim_updates))


def _with_objective(
    config: ExperimentConfig, **objective_updates: object
) -> ExperimentConfig:
    return replace(config, objective=replace(config.objective, **objective_updates))


def _l2_debug_config(
    name: str, lambda_sigreg: float, sigreg_warmup_steps: int
) -> ExperimentConfig:
    return _with_data(
        _with_objective(
            _with_runtime(
                replace(
                    _base_experiment(name, "l2_sigreg", lambda_sigreg),
                    model=_l2_model_config(),
                ),
                train_steps=200,
                eval_every=20,
                checkpoint_every=100,
            ),
            num_slices=256,
            sigreg_warmup_steps=sigreg_warmup_steps,
        ),
        batch_size=8,
        eval_batch_size=8,
        train_token_budget=65_536,
        val_token_budget=8_192,
        max_train_documents=512,
        max_val_documents=128,
    )


def _l2_debug_ablation_config(
    name: str,
    *,
    learned_output_scale: bool,
    sigreg_random_vocab_size: int,
) -> ExperimentConfig:
    return _with_data(
        _with_objective(
            _with_runtime(
                replace(
                    _base_experiment(name, "l2_sigreg", 1.0),
                    model=_l2_model_config(),
                ),
                train_steps=200,
                eval_every=20,
                checkpoint_every=100,
            ),
            num_slices=256,
            sigreg_warmup_steps=0,
            learned_output_scale=learned_output_scale,
            sigreg_random_vocab_size=sigreg_random_vocab_size,
        ),
        batch_size=8,
        eval_batch_size=8,
        train_token_budget=65_536,
        val_token_budget=8_192,
        max_train_documents=512,
        max_val_documents=128,
    )


def _l2_midrun_config(
    name: str,
    *,
    learned_output_scale: bool,
    sigreg_random_vocab_size: int,
) -> ExperimentConfig:
    return _with_data(
        _with_objective(
            _with_runtime(
                replace(
                    _base_experiment(name, "l2_sigreg", 1.0),
                    model=_l2_model_config(),
                ),
                train_steps=1_000,
                eval_every=100,
                checkpoint_every=250,
            ),
            num_slices=256,
            sigreg_warmup_steps=0,
            learned_output_scale=learned_output_scale,
            sigreg_random_vocab_size=sigreg_random_vocab_size,
        ),
        batch_size=8,
        eval_batch_size=8,
        train_token_budget=131_072,
        val_token_budget=8_192,
        max_train_documents=1_024,
        max_val_documents=128,
    )


def _l2_vocab_sigreg_debug_config(
    name: str, *, embedding_init_std: float
) -> ExperimentConfig:
    return _with_data(
        _with_objective(
            _with_optim(
                _with_runtime(
                    replace(
                        _base_experiment(name, "l2_sigreg", 0.05),
                        model=_l2_model_config_with_embedding_std(embedding_init_std),
                    ),
                    train_steps=200,
                    eval_every=20,
                    checkpoint_every=100,
                ),
                l2_embedding_lr_scale=0.1,
            ),
            num_slices=256,
            sigreg_warmup_steps=0,
            learned_output_scale=False,
            sigreg_random_vocab_size=2_048,
            sigreg_include_active_predictions=False,
        ),
        batch_size=8,
        eval_batch_size=8,
        train_token_budget=65_536,
        val_token_budget=8_192,
        max_train_documents=512,
        max_val_documents=128,
    )


def _l2_debug_nowarm_init_config(
    name: str, embedding_init_std: float
) -> ExperimentConfig:
    return _with_data(
        _with_objective(
            _with_runtime(
                replace(
                    _base_experiment(name, "l2_sigreg", 1.0),
                    model=_l2_model_config_with_embedding_std(embedding_init_std),
                ),
                train_steps=200,
                eval_every=20,
                checkpoint_every=100,
            ),
            num_slices=256,
            sigreg_warmup_steps=0,
        ),
        batch_size=8,
        eval_batch_size=8,
        train_token_budget=65_536,
        val_token_budget=8_192,
        max_train_documents=512,
        max_val_documents=128,
    )


def _l2_longrun_config(name: str, train_steps: int) -> ExperimentConfig:
    return _with_data(
        _with_runtime(
            replace(
                _base_experiment(name, "l2_sigreg", 1.0),
                model=_l2_model_config(),
            ),
            train_steps=train_steps,
            eval_every=250,
            checkpoint_every=1000,
        ),
        batch_size=8,
        eval_batch_size=8,
        train_token_budget=524_288,
        val_token_budget=16_384,
        max_train_documents=4_096,
        max_val_documents=256,
    )


def get_config(name: str) -> ExperimentConfig:
    presets = {
        "l2_debug": _l2_debug_config("l2_debug", 1.0, 100),
        "l2_debug_nowarm": _l2_debug_config("l2_debug_nowarm", 1.0, 0),
        "l2_debug_scale_only": _l2_debug_ablation_config(
            "l2_debug_scale_only",
            learned_output_scale=True,
            sigreg_random_vocab_size=0,
        ),
        "l2_debug_sigmix_512": _l2_debug_ablation_config(
            "l2_debug_sigmix_512",
            learned_output_scale=False,
            sigreg_random_vocab_size=512,
        ),
        "l2_debug_sigmix_only": _l2_debug_ablation_config(
            "l2_debug_sigmix_only",
            learned_output_scale=False,
            sigreg_random_vocab_size=2_048,
        ),
        "l2_debug_scale_sigmix": _l2_debug_ablation_config(
            "l2_debug_scale_sigmix",
            learned_output_scale=True,
            sigreg_random_vocab_size=2_048,
        ),
        "l2_debug_vocabsig_0p01": _l2_vocab_sigreg_debug_config(
            "l2_debug_vocabsig_0p01",
            embedding_init_std=0.01,
        ),
        "l2_debug_vocabsig_0p8": _l2_vocab_sigreg_debug_config(
            "l2_debug_vocabsig_0p8",
            embedding_init_std=0.8,
        ),
        "l2_midrun_scale_sigmix": _l2_midrun_config(
            "l2_midrun_scale_sigmix",
            learned_output_scale=True,
            sigreg_random_vocab_size=2_048,
        ),
        "l2_1h": _l2_longrun_config("l2_1h", 8_500),
        "l2_debug_nowarm_lam_0p1": _l2_debug_config("l2_debug_nowarm_lam_0p1", 0.1, 0),
        "l2_debug_nowarm_lam_0p3": _l2_debug_config("l2_debug_nowarm_lam_0p3", 0.3, 0),
        "l2_debug_nowarm_lam_0p6": _l2_debug_config("l2_debug_nowarm_lam_0p6", 0.6, 0),
        "l2_debug_nowarm_lam_1p0": _l2_debug_config("l2_debug_nowarm_lam_1p0", 1.0, 0),
        "l2_debug_nowarm_lam_1p5": _l2_debug_config("l2_debug_nowarm_lam_1p5", 1.5, 0),
        "l2_debug_nowarm_lam_2p0": _l2_debug_config("l2_debug_nowarm_lam_2p0", 2.0, 0),
        "l2_debug_nowarm_init_0p000": _l2_debug_nowarm_init_config(
            "l2_debug_nowarm_init_0p000", 0.0
        ),
        "l2_debug_nowarm_init_0p004": _l2_debug_nowarm_init_config(
            "l2_debug_nowarm_init_0p004", 0.004
        ),
        "l2_debug_nowarm_init_0p008": _l2_debug_nowarm_init_config(
            "l2_debug_nowarm_init_0p008", 0.008
        ),
        "l2_debug_nowarm_init_0p012": _l2_debug_nowarm_init_config(
            "l2_debug_nowarm_init_0p012", 0.012
        ),
        "l2_debug_nowarm_init_0p016": _l2_debug_nowarm_init_config(
            "l2_debug_nowarm_init_0p016", 0.016
        ),
        "ce_debug": _with_data(
            _with_runtime(
                _base_experiment("ce_debug", "ce_baseline", 0.0),
                train_steps=200,
                eval_every=20,
                checkpoint_every=100,
            ),
            batch_size=8,
            eval_batch_size=8,
            train_token_budget=65_536,
            val_token_budget=8_192,
            max_train_documents=512,
            max_val_documents=128,
        ),
        "l2_smoke": _with_data(
            _with_objective(
                _with_optim(
                    _with_runtime(
                        replace(
                            _base_experiment("l2_smoke", "l2_sigreg", 0.02),
                            model=_l2_model_config(),
                        ),
                        train_steps=10,
                        eval_every=5,
                        checkpoint_every=10,
                    ),
                    decay_frac=0.0,
                ),
                num_slices=64,
                sigreg_warmup_steps=1_000,
            ),
            batch_size=4,
            eval_batch_size=4,
            train_token_budget=16_384,
            val_token_budget=4_096,
            max_train_documents=128,
            max_val_documents=64,
        ),
        "ce_smoke": _with_data(
            _with_optim(
                _with_runtime(
                    _base_experiment("ce_smoke", "ce_baseline", 0.0),
                    train_steps=10,
                    eval_every=5,
                    checkpoint_every=10,
                ),
                decay_frac=0.0,
            ),
            batch_size=4,
            eval_batch_size=4,
            train_token_budget=16_384,
            val_token_budget=4_096,
            max_train_documents=128,
            max_val_documents=64,
        ),
    }
    if name not in presets:
        available = ", ".join(sorted(presets))
        raise KeyError(f"Unknown config '{name}'. Available: {available}")
    return presets[name]
