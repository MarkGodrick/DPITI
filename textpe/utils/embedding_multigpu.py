import pandas as pd
import multiprocessing as mp
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


class multigpu_hfpipe_embedding(Embedding):
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
        self._batch_size = batch_size

        self._temp_folder = tempfile.TemporaryDirectory()
        self._inception = InceptionV3W(path="/data/whx/models", download=True, resize_inside=False).to("cuda")
        self._resize_pre = make_resizer("PIL", quantize_after=False, filter="bicubic", output_size=(256, 256))
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
            f"Embedding: computing {self.column_name} for {len(uncomputed_data.data_frame)}/{len(data.data_frame)} samples"
        )
        samples = uncomputed_data.data_frame[TEXT_DATA_COLUMN_NAME].tolist()

        # optional: extract "content" part
        pattern = r'([^:]*:|^)(.*?)(?=\.$|$)'
        matches = [re.search(pattern, str(text), re.DOTALL) for text in samples]
        samples = [match.group(2).strip() for match in matches]

        images = self._generate_images_multi_gpu(samples)

        if images[0].mode != "RGB":
            images = [img.convert("RGB") for img in images]
        images = np.stack([np.array(img) for img in images], axis=0)

        if images.shape[3] == 1:
            images = np.repeat(images, 3, axis=3)

        # compute embeddings using InceptionV3
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
            f"Embedding: finished computing {self.column_name} for {len(uncomputed_data.data_frame)}/{len(data.data_frame)} samples"
        )
        return self.merge_computed_rows(data, uncomputed_data)

    # === Internal Multi-GPU Generate Functions ===
    def _generate_images_multi_gpu(self, all_prompts):

        def generate_on_device(device_id, prompts, batch_size, queue, model_name, start_idx):
            torch.cuda.set_device(device_id)
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            ).to(f"cuda:{device_id}")

            results = []
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                images = pipe(batch_prompts, num_inference_steps=4, guidance_scale=0.0).images
                results.extend(images)

            queue.put((start_idx, results))

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices found.")

        prompts_split = np.array_split(all_prompts, num_gpus)
        queue = mp.Queue()
        processes = []

        # 为每段记录起始 index
        start_indices = np.cumsum([0] + [len(p) for p in prompts_split[:-1]])

        for device_id in range(num_gpus):
            p = mp.Process(
                target=generate_on_device,
                args=(
                    device_id,
                    prompts_split[device_id].tolist(),
                    self._batch_size,
                    queue,
                    self._model_name,
                    start_indices[device_id]
                )
            )
            p.start()
            processes.append(p)

        # 初始化空列表存储所有结果
        total_len = len(all_prompts)
        all_images = [None] * total_len

        for _ in range(num_gpus):
            start_idx, images = queue.get()
            all_images[start_idx:start_idx + len(images)] = images

        for p in processes:
            p.join()

        return all_images


class multigpu_infinity_embedding(Embedding):
    def __init__(self, config, batch_size=4):
        super().__init__()
        self.config = config
        self._model_name = config.model_type

        # not loading model here
        self._batch_size = batch_size

        self._temp_folder = tempfile.TemporaryDirectory()
        self._inception = InceptionV3W(path="/data/whx/models", download=True, resize_inside=False).to("cuda")
        self._resize_pre = make_resizer("PIL", quantize_after=False, filter="bicubic", output_size=(256, 256))
        self._resizer = build_resizer("clean")

    @property
    def column_name(self):
        return f"{EMBEDDING_COLUMN_NAME}.{type(self).__name__}.{self._model_name}"

    def compute_embedding(self, data):
        uncomputed_data = self.filter_uncomputed_rows(data)
        if len(uncomputed_data.data_frame) == 0:
            execution_logger.info(f"Embedding: {self.column_name} already computed")
            return data

        execution_logger.info(
            f"Embedding: computing {self.column_name} for {len(uncomputed_data.data_frame)}/{len(data.data_frame)} samples"
        )
        samples = uncomputed_data.data_frame[TEXT_DATA_COLUMN_NAME].tolist()

        # 正则过滤
        pattern = r'([^:]*:|^)(.*?)(?=\.$|$)'
        matches = [re.search(pattern, str(text), re.DOTALL) for text in samples]
        samples = [match.group(2).strip() for match in matches]

        images = self._generate_images_multi_gpu(samples)
        images = np.stack(images, axis=0)

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

    def _generate_images_multi_gpu(self, all_prompts):

        def generate_on_device(device_id, prompts, queue, config, start_idx):
            
            torch.cuda.set_device(device_id)
            
            text_tokenizer, text_encoder = load_tokenizer(t5_path=config.text_encoder_ckpt)
            vae = load_visual_tokenizer(config)
            infinity = load_transformer(vae, config)

            cfg = 3
            tau = 0.5
            h_div_w = 1/1
            seed = random.randint(0, 10000)
            enable_positive_prompt = 0

            h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
            scale_schedule = dynamic_resolution_h_w[h_div_w_template_][config.pn]['scales']
            scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

            result_images = []
            for prompt in prompts:
                image = gen_one_img(
                    infinity, vae, text_tokenizer, text_encoder, prompt,
                    g_seed=seed,
                    gt_leak=0,
                    gt_ls_Bl=None,
                    cfg_list=cfg,
                    tau_list=tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[config.cfg_insertion_layer],
                    vae_type=config.vae_type,
                    sampling_per_bits=config.sampling_per_bits,
                    enable_positive_prompt=enable_positive_prompt,
                )
                result_images.append(image.cpu().numpy())
            queue.put((start_idx, result_images))

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices found.")

        prompts_split = np.array_split(all_prompts, num_gpus)
        start_indices = np.cumsum([0] + [len(p) for p in prompts_split[:-1]])
        queue = mp.Queue()
        processes = []

        for device_id in range(num_gpus):
            p = mp.Process(
                target=generate_on_device,
                args=(device_id, prompts_split[device_id].tolist(), queue, self.config, start_indices[device_id])
            )
            p.start()
            processes.append(p)

        # 收集图像并按顺序插入
        total_len = len(all_prompts)
        all_images = [None] * total_len
        for _ in range(num_gpus):
            start_idx, images = queue.get()
            all_images[start_idx:start_idx + len(images)] = images

        for p in processes:
            p.join()

        return all_images
