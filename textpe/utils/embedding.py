import pandas as pd
import tempfile
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from cleanfid.inception_torchscript import InceptionV3W
from cleanfid.resize import build_resizer
from cleanfid.resize import make_resizer

from pe.embedding import Embedding
from pe.logging import execution_logger
from pe.constant.data import TEXT_DATA_COLUMN_NAME
from pe.constant.data import EMBEDDING_COLUMN_NAME

from diffusers import StableDiffusionXLPipeline
import re

from DPLDM.ldm.util import instantiate_from_config
from DPLDM.ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError

import cv2
from Infinity.tools.run_infinity import *
import random

def to_uint8(x, min, max):
    x = (x - min) / (max - min)
    x = np.around(np.clip(x * 255, a_min=0, a_max=255)).astype(np.uint8)
    return x

def load_model_from_config(config, ckpt):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    try:
        config.model.params.ignore_keys = []
        config.model.params.ckpt_path = None
    except ConfigAttributeError:
        pass
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


class hfpipe_embedding(Embedding):
    """Compute the embeddings of text using huggingface."""

    def __init__(self, model, batch_size=4):
        """Constructor.

        :param model: The Sentence Transformers model to use
        :type model: str
        :param batch_size: The batch size to use for computing the embedding, defaults to 2000
        :type batch_size: int, optional
        """
        super().__init__()
        self._model_name = model
        self._pipe = StableDiffusionXLPipeline.from_pretrained(self._model_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
        self._batch_size = batch_size

        self._temp_folder = tempfile.TemporaryDirectory()
        self._inception = InceptionV3W(path="/data/whx/models", download=True, resize_inside=False).to("cuda")
        self._resize_pre = make_resizer(
            library="PIL",
            quantize_after=False,
            filter="bicubic",
            output_size=(256, 256),
        )
        self._resizer = build_resizer("clean")

    @property
    def column_name(self):
        """The column name to be used in the data frame."""
        return f"{EMBEDDING_COLUMN_NAME}.{type(self).__name__}.{self._model_name}"

    def compute_embedding(self, data):
        """Compute the Sentence Transformers embedding of text.

        :param data: The data object containing the text
        :type data: :py:class:`pe.data.Data`
        :return: The data object with the computed embedding
        :rtype: :py:class:`pe.data.Data`
        """
        uncomputed_data = self.filter_uncomputed_rows(data)
        if len(uncomputed_data.data_frame) == 0:
            execution_logger.info(f"Embedding: {self.column_name} already computed")
            return data
        execution_logger.info(
            f"Embedding: computing {self.column_name} for {len(uncomputed_data.data_frame)}/{len(data.data_frame)}"
            " samples"
        )
        samples = uncomputed_data.data_frame[TEXT_DATA_COLUMN_NAME].tolist()
        # embeddings = self._model.encode(samples, batch_size=self._batch_size)

        # do sample filter
        pattern = r'([^:]*:|^)(.*?)(?=\.$|$)'
        matches = [re.search(pattern,str(text),re.DOTALL) for text in samples]

        samples = [match.group(2).strip() for match in matches]

        # generate images from sample texts
        images = []
        for batch_idx in tqdm(range((len(samples)+self._batch_size-1)//self._batch_size)):
            images.append(self._pipe(samples[batch_idx*self._batch_size:(batch_idx+1)*self._batch_size], num_inference_steps=4,guidance_scale=0.0).images)
        images = np.concatenate(images,axis=0)

        # compute embedding using InceptionV3
        if images.shape[3] == 1:
            images = np.repeat(images, 3, axis=3)
        embeddings = []
        for i in tqdm(range(0, len(images), self._batch_size)):
            transformed_x = []
            for j in range(i, min(i + self._batch_size, len(images))):
                image = images[j]
                image = self._resize_pre(image)
                image = to_uint8(image, min=0, max=255)
                image = self._resizer(image)
                transformed_x.append(image)
            transformed_x = np.stack(transformed_x, axis=0).transpose((0, 3, 1, 2))
            embeddings.append(self._inception(torch.from_numpy(transformed_x).to("cuda")))
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.cpu().detach().numpy()

        uncomputed_data.data_frame[self.column_name] = pd.Series(
            list(embeddings), index=uncomputed_data.data_frame.index
        )
        execution_logger.info(
            f"Embedding: finished computing {self.column_name} for "
            f"{len(uncomputed_data.data_frame)}/{len(data.data_frame)} samples"
        )
        return self.merge_computed_rows(data, uncomputed_data)



class dpldm_embedding(Embedding):
    """Compute the embedding of text using dpldm models."""

    def __init__(self, config_path, ckpt_path, num_sample_steps = 200, batch_size= 64 ,eta = 1.0):
        """Constructor.

        :param config_path: path to the config.yaml file
        :type config_path: str
        :param ckpt_path: path to the ckpt.pt file
        :type ckpt_path: str
        :param batch_size: The batch size to use for computing the embedding, defaults to 2000
        :type batch_size: int, optional
        """
        super().__init__()
        self._model_name = "DPLDM_txt2img"
        self._config_path = config_path
        self._ckpt_path = ckpt_path
        self.config = OmegaConf.load(self._config_path)
        model = load_model_from_config(self.config, self._ckpt_path)

        self._pipe = DDIMSampler(model)
        self._num_sample_steps = num_sample_steps
        self._batch_size = batch_size
        self._shape = (model.model.diffusion_model.in_channels,
                       model.model.diffusion_model.image_size,
                       model.model.diffusion_model.image_size)
        self._eta = eta

        self._temp_folder = tempfile.TemporaryDirectory()
        self._inception = InceptionV3W(path=self._temp_folder.name, download=True, resize_inside=False).to("cuda")
        self._resize_pre = make_resizer(
            library="PIL",
            quantize_after=False,
            filter="bicubic",
            output_size=(256, 256),
        )
        self._resizer = build_resizer("clean")

    @property
    def column_name(self):
        """The column name to be used in the data frame."""
        return f"{EMBEDDING_COLUMN_NAME}.{type(self).__name__}.{self._model_name}"

    def compute_embedding(self, data):
        """Compute the Sentence Transformers embedding of text.

        :param data: The data object containing the text
        :type data: :py:class:`pe.data.Data`
        :return: The data object with the computed embedding
        :rtype: :py:class:`pe.data.Data`
        """
        uncomputed_data = self.filter_uncomputed_rows(data)
        if len(uncomputed_data.data_frame) == 0:
            execution_logger.info(f"Embedding: {self.column_name} already computed")
            return data
        execution_logger.info(
            f"Embedding: computing {self.column_name} for {len(uncomputed_data.data_frame)}/{len(data.data_frame)}"
            " samples"
        )
        samples = uncomputed_data.data_frame[TEXT_DATA_COLUMN_NAME].tolist()
        # embeddings = self._model.encode(samples, batch_size=self._batch_size)

        # do sample filter
        pattern = r'([^:]*:|^)(.*?)(?=\.$|$)'
        matches = [re.search(pattern,str(text),re.DOTALL) for text in samples]

        samples = [match.group(2).strip() for match in matches]

        # generate images from sample texts
        images = []
        for batch_idx in tqdm(range((len(samples)+self._batch_size-1)//self._batch_size)):
            batch_samples = samples[batch_idx*self._batch_size:(batch_idx+1)*self._batch_size]
            batch_conditioning = self._pipe.model.get_learned_conditioning(batch_samples)
            sample_images, _ = self._pipe.sample(
                S = self._num_sample_steps,
                batch_size = min(len(batch_samples),self._batch_size),
                shape = self._shape,
                conditioning = batch_conditioning,
                verbose = False,
                eta = self._eta
            )
            batch_images = self._pipe.model.decode_first_stage(sample_images).cpu().numpy()
            batch_images = np.transpose(batch_images,(0,2,3,1))
            batch_images = (batch_images + 1) * 127.5
            batch_images = batch_images.clip(0,255).astype(np.uint8)
            images.append(batch_images)
        images = np.concatenate(images,axis=0)

        assert images.ndim==4
        # compute embedding using InceptionV3
        if images.shape[3] == 1:
            images = np.repeat(images, 3, axis=3)
        embeddings = []
        for i in tqdm(range(0, len(images), self._batch_size)):
            transformed_x = []
            for j in range(i, min(i + self._batch_size, len(images))):
                image = images[j]
                image = self._resize_pre(image)
                image = to_uint8(image, min=0, max=255)
                image = self._resizer(image)
                transformed_x.append(image)
            transformed_x = np.stack(transformed_x, axis=0).transpose((0, 3, 1, 2))
            embeddings.append(self._inception(torch.from_numpy(transformed_x).to("cuda")))
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.cpu().detach().numpy()

        uncomputed_data.data_frame[self.column_name] = pd.Series(
            list(embeddings), index=uncomputed_data.data_frame.index
        )
        execution_logger.info(
            f"Embedding: finished computing {self.column_name} for "
            f"{len(uncomputed_data.data_frame)}/{len(data.data_frame)} samples"
        )
        return self.merge_computed_rows(data, uncomputed_data)

class infinity_embedding(Embedding):
    """Compute the embeddings of text using huggingface."""

    def __init__(self, config, batch_size = 4):
        """Constructor.
        TODO
        """
        super().__init__()
        self.config = config
        self._model_name = config.model_type
        # load text encoder
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=self.config.text_encoder_ckpt)
        # load vae
        self.vae = load_visual_tokenizer(self.config)
        # load infinity
        self.infinity = load_transformer(self.vae, self.config)

        self._batch_size = batch_size

        self._temp_folder = tempfile.TemporaryDirectory()
        self._inception = InceptionV3W(path=self._temp_folder.name, download=True, resize_inside=False).to("cuda")
        self._resize_pre = make_resizer(
            library="PIL",
            quantize_after=False,
            filter="bicubic",
            output_size=(256, 256),
        )
        self._resizer = build_resizer("clean")

    @property
    def column_name(self):
        """The column name to be used in the data frame."""
        return f"{EMBEDDING_COLUMN_NAME}.{type(self).__name__}.{self._model_name}"

    def compute_embedding(self, data):
        """Compute the Sentence Transformers embedding of text.

        :param data: The data object containing the text
        :type data: :py:class:`pe.data.Data`
        :return: The data object with the computed embedding
        :rtype: :py:class:`pe.data.Data`
        """
        uncomputed_data = self.filter_uncomputed_rows(data)
        if len(uncomputed_data.data_frame) == 0:
            execution_logger.info(f"Embedding: {self.column_name} already computed")
            return data
        execution_logger.info(
            f"Embedding: computing {self.column_name} for {len(uncomputed_data.data_frame)}/{len(data.data_frame)}"
            " samples"
        )
        samples = uncomputed_data.data_frame[TEXT_DATA_COLUMN_NAME].tolist()
        # embeddings = self._model.encode(samples, batch_size=self._batch_size)

        # do sample filter
        pattern = r'([^:]*:|^)(.*?)(?=\.$|$)'
        matches = [re.search(pattern,str(text),re.DOTALL) for text in samples]
        samples = [match.group(2).strip() for match in matches]

        # generate images from sample texts
        cfg = 3
        tau = 0.5
        h_div_w = 1/1 # aspect ratio, height:width
        seed = random.randint(0, 10000)
        enable_positive_prompt=0

        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][self.config.pn]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
        images = []
        for batch_idx in tqdm(range((len(samples)+self._batch_size-1)//self._batch_size)):
            for idx in range(batch_idx*self._batch_size,(batch_idx+1)*self._batch_size):
                image = gen_one_img(
                    self.infinity,
                    self.vae,
                    self.text_tokenizer,
                    self.text_encoder,
                    samples[idx],
                    g_seed=seed,
                    gt_leak=0,
                    gt_ls_Bl=None,
                    cfg_list=cfg,
                    tau_list=tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[self.config.cfg_insertion_layer],
                    vae_type=self.config.vae_type,
                    sampling_per_bits=self.config.sampling_per_bits,
                    enable_positive_prompt=enable_positive_prompt,
                )
                images.append(image.cpu().numpy())
        images = np.stack(images,axis=0)

        # compute embedding using InceptionV3
        if images.shape[3] == 1:
            images = np.repeat(images, 3, axis=3)
        embeddings = []
        for i in tqdm(range(0, len(images), self._batch_size)):
            transformed_x = []
            for j in range(i, min(i + self._batch_size, len(images))):
                image = images[j]
                image = self._resize_pre(image)
                image = self._resizer(image)
                transformed_x.append(image)
            transformed_x = np.stack(transformed_x, axis=0).transpose((0, 3, 1, 2))
            embeddings.append(self._inception(torch.from_numpy(transformed_x).to("cuda")))
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.cpu().detach().numpy()

        uncomputed_data.data_frame[self.column_name] = pd.Series(
            list(embeddings), index=uncomputed_data.data_frame.index
        )
        execution_logger.info(
            f"Embedding: finished computing {self.column_name} for "
            f"{len(uncomputed_data.data_frame)}/{len(data.data_frame)} samples"
        )
        return self.merge_computed_rows(data, uncomputed_data)