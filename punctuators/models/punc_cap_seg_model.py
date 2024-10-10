import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from sentencepiece import SentencePieceProcessor
from torch.utils.data import DataLoader
import numpy as np
from punctuators.collectors import PunctCapSegResultCollector
from punctuators.data import TextInferenceDataset

@dataclass
class PunctCapSegConfig:
    """ No abstraction yet, will be refactored """

    pass


@dataclass
class PunctCapSegConfigONNX(PunctCapSegConfig):
    spe_filename: str = "sp.model"
    model_filename: str = "model.onnx"
    config_filename: str = "config.yaml"
    hf_repo_id: Optional[str] = None
    directory: Optional[str] = None


class PunctCapSegModel:
    """ No abstraction yet, will be refactored """

    def __init__(self):
        pass


class PunctCapSegModelONNX(PunctCapSegModel):
    def __init__(self, cfg: PunctCapSegConfigONNX, ort_providers: Optional[Any]=None, model_path = None):
        super().__init__()
        if cfg.hf_repo_id is not None:
            self._spe_path = hf_hub_download(repo_id=cfg.hf_repo_id, filename=cfg.spe_filename)
            print(f'Model path = {model_path}')
            if model_path:
                onnx_path = model_path
            else:
                onnx_path = hf_hub_download(repo_id=cfg.hf_repo_id, filename=cfg.model_filename)
            config_path = hf_hub_download(repo_id=cfg.hf_repo_id, filename=cfg.config_filename)
        else:
            if cfg.directory is None:
                raise ValueError(f"Need HF repo ID or local directory name; got {cfg}")
            self._spe_path = os.path.join(cfg.directory, cfg.spe_filename)
            onnx_path = os.path.join(cfg.directory, cfg.model_filename)
            config_path = os.path.join(cfg.directory, cfg.config_filename)
        self._tokenizer: SentencePieceProcessor = SentencePieceProcessor(self._spe_path)  # noqa
        if not ort_providers:
            ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._ort_session: ort.InferenceSession = ort.InferenceSession(
            onnx_path, providers=ort_providers
        )
        self._config = OmegaConf.load(config_path)
        self._max_len = self._config.max_length
        self._pre_labels: List[str] = self._config.pre_labels
        self._post_labels: List[str] = self._config.post_labels
        self._languages: List[str] = self._config.languages
        self._null_token = self._config.get("null_token", "<NULL>")

    @classmethod
    def pretrained_model_info(cls) -> Dict[str, PunctCapSegConfigONNX]:
        info = {
            "pcs_47lang": PunctCapSegConfigONNX(
                hf_repo_id="1-800-BAD-CODE/punct_cap_seg_47_language",
                spe_filename="spe_unigram_64k_lowercase_47lang.model",
                model_filename="punct_cap_seg_47lang.onnx",
            ),
            "pcs_en": PunctCapSegConfigONNX(
                hf_repo_id="1-800-BAD-CODE/punct_cap_seg_en",
                spe_filename="spe_32k_lc_en.model",
                model_filename="punct_cap_seg_en.onnx",
            ),
            "pcs_romance": PunctCapSegConfigONNX(
                hf_repo_id="1-800-BAD-CODE/punctuation_fullstop_truecase_romance",
                spe_filename="sp.model",
                model_filename="model.onnx",
            ),
        }
        return info

    @classmethod
    def from_pretrained(cls, pretrained_name: str, ort_providers: Optional[Any]=None, model_path=None) -> "PunctCapSegModelONNX":
        """

        Args:
            pretrained_name: 
        """
        cfg: PunctCapSegConfigONNX
        if "/" in pretrained_name:
            # Assume this is a HuggingFace repository with default model names
            cfg = PunctCapSegConfigONNX(hf_repo_id=pretrained_name)
        else:
            available_models: Dict[str, PunctCapSegConfigONNX] = cls.pretrained_model_info()
            if pretrained_name not in available_models:
                raise ValueError(
                    f"Pretrained name '{pretrained_name}' not in available models: '{list(available_models.keys())}'"
                )
            cfg = available_models[pretrained_name]
        return cls(cfg=cfg, ort_providers=ort_providers, model_path=model_path)

    @property
    def languages(self) -> List[str]:
        return self._languages

    def _setup_dataloader(
        self, texts: List[str], batch_size_tokens: int, overlap: int, num_workers: int = os.cpu_count() - 1
    ) -> DataLoader:
        dataset: TextInferenceDataset = TextInferenceDataset(
            texts=texts,
            batch_size_tokens=batch_size_tokens,
            overlap=overlap,
            max_length=self._max_len,
            spe_model_path=self._spe_path,
            spe_model=self._tokenizer
        )
        return DataLoader(
            dataset=dataset, num_workers=num_workers, collate_fn=dataset.collate_fn, batch_sampler=dataset.sampler,
        )

    @torch.inference_mode()
    @torch.amp.autocast('cuda', cache_enabled=False, dtype=torch.bfloat16)
    def infer(
        self,
        texts: List[str],
        apply_sbd: bool = True,
        batch_size_tokens: int = 5460,
        overlap: int = 16,
        num_workers: int = 0,
    ) -> Union[List[str], List[List[str]]]:
        collectors: List[PunctCapSegResultCollector] = [
            PunctCapSegResultCollector(sp_model=self._tokenizer, apply_sbd=apply_sbd, overlap=overlap)
            for _ in range(len(texts))
        ]
        dataloader: DataLoader = self._setup_dataloader(
            texts=texts, batch_size_tokens=batch_size_tokens, overlap=overlap, num_workers=num_workers
        )
        batch: Tuple[torch.Tensor, ...]
        for batch in dataloader:
            input_ids, batch_indices, input_indices, lengths = batch
            # Get predictions.
            pre_preds, post_preds, cap_preds, seg_preds = self._ort_session.run(None, {"input_ids": input_ids.numpy()})
            batch_size = input_ids.shape[0]
            lengths = lengths.tolist()
            batch_indices = batch_indices.tolist()
            input_indices = input_indices.tolist()

            # Extract segments (batch processing)
            segment_ids = [input_ids[i, 1 : lengths[i] - 1].tolist() for i in range(batch_size)]
            segment_pre_preds = [pre_preds[i, 1 : lengths[i] - 1].tolist() for i in range(batch_size)]
            segment_post_preds = [post_preds[i, 1 : lengths[i] - 1].tolist() for i in range(batch_size)]
            segment_cap_preds = [cap_preds[i, 1 : lengths[i] - 1].tolist() for i in range(batch_size)]
            segment_sbd_preds = [seg_preds[i, 1 : lengths[i] - 1].tolist() for i in range(batch_size)]

            # Vectorized token replacement using numpy
            pre_labels_array = np.array(self._pre_labels)
            pre_tokens = [pre_labels_array[segment].tolist() for segment in segment_pre_preds]
            post_labels_array = np.array(self._post_labels)
            post_tokens = [post_labels_array[segment].tolist() for segment in segment_post_preds]

            # Replace null tokens (Vectorized)
            pre_tokens = np.array(pre_tokens, dtype=object)
            post_tokens = np.array(post_tokens, dtype=object)
            pre_tokens[pre_tokens == self._null_token] = None
            post_tokens[post_tokens == self._null_token] = None

            # Use a bulk collect method (if possible, otherwise process in chunks)
            for i in range(batch_size):
                collectors[batch_indices[i]].collect(
                    ids=segment_ids[i],
                    pre_preds=pre_tokens[i],
                    post_preds=post_tokens[i],
                    cap_preds=segment_cap_preds[i],
                    sbd_preds=segment_sbd_preds[i],
                    idx=input_indices[i],
                )

        outputs: Union[List[str], List[List[str]]] = [x.produce() for x in collectors]
        return outputs
