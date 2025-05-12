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

from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
import re

from DPLDM.ldm.util import instantiate_from_config
from DPLDM.ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
import threading

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

# --- Worker function for each thread ---
# This function will be executed by a separate thread and assigned to a specific GPU
def generate_and_embed_worker(
    gpu_id,
    prompts_chunk,
    original_indices_chunk, # Pass original indices to map results back correctly
    model_name,
    pipeline,
    sample_config,
    batch_size,
    results_list, # List to store results from this thread
    worker_id # For logging
):
    """
    Worker thread function to generate images and compute embeddings on a specific GPU.
    """
    try:
        # Set the CUDA device for this thread
        torch.cuda.set_device(gpu_id)
        execution_logger.debug(f"Worker {worker_id} on GPU {gpu_id} starting with {len(prompts_chunk)} prompts.")

        # Move models to the designated GPU
        # Important: If pipe/inception are shared objects passed to threads,
        # their internal state (like device) might need careful handling.
        # Moving them *after* setting device is common practice.
        pipe = pipeline.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(gpu_id)
        
        
        temp_folder = tempfile.TemporaryDirectory()
        inception = InceptionV3W(path=temp_folder.name, download=True, resize_inside=False)
        resize_pre = make_resizer(
            library="PIL",
            quantize_after=False,
            filter="bicubic",
            output_size=(256, 256),
        )
        resizer = build_resizer("clean")
        
        inception.to(gpu_id)
        # Ensure models are in evaluation mode
        inception.eval()


        # --- Generate Images ---
        local_generated_images_pil = []
        pbar_gen = tqdm(range(0, len(prompts_chunk), batch_size),
                        desc=f"Generating (Worker {worker_id}, GPU {gpu_id})",
                        leave=False) # Use leave=False to avoid cluttering terminal

        for i in pbar_gen:
            batch_prompts = prompts_chunk[i:i+batch_size]
            with torch.no_grad():
                images_batch_pil = pipe(batch_prompts, **sample_config).images
            local_generated_images_pil.extend(images_batch_pil)

        execution_logger.debug(f"Worker {worker_id} generated {len(local_generated_images_pil)} images.")

        # --- Compute Inception Embeddings ---
        embeddings_list = []
        if len(local_generated_images_pil) > 0:
            # Convert list of PIL images to a NumPy array (N, H, W, C)
            # Assuming SDXL generates fixed size images, stacking is straightforward
            images_np = np.stack([np.array(img) for img in local_generated_images_pil], axis=0)

            # Ensure images are in the correct format (uint8 HWC) and channel count (3) for resizer/inception
            # Assuming images_np is uint8 HWC [0, 255] from PIL conversion
            if images_np.dtype != np.uint8:
                # Basic scaling if needed (e.g., from float [0, 1] to uint8 [0, 255])
                # Adjust this based on the actual output format of your pipeline images
                images_np = (images_np * 255).astype(np.uint8) # Example scaling


            if images_np.shape[-1] == 1:
                images_np = np.repeat(images_np, 3, axis=-1) # Use axis=-1 for the last dimension


            inception_batch_size = batch_size # Or a separate config

            pbar_inception = tqdm(range(0, len(images_np), inception_batch_size),
                                  desc=f"Inception (Worker {worker_id}, GPU {gpu_id})",
                                  leave=False)

            for i in pbar_inception:
                batch_np = images_np[i : i + inception_batch_size] # NumPy batch (N, H, W, C) uint8

                # Apply preprocessing steps
                processed_batch_np = []
                for img_np in batch_np:
                    # img_np is (H, W, C) uint8
                    img_np_pre = resize_pre(img_np) # Apply initial resize if needed
                    # to_uint8 might be redundant if img_np_pre is already uint8, confirm its use
                    # img_uint8 = to_uint8(img_np_pre, min=0, max=255) # Original line
                    img_resized_np = resizer(img_np_pre) # Apply cleanfid resizer (returns numpy HWC, potentially float/standardized)
                    processed_batch_np.append(img_resized_np)

                # Stack processed numpy images and convert to torch tensor (N, C, H, W)
                # Assuming resizer returns float HWC
                processed_batch_np = np.stack(processed_batch_np, axis=0) # Stack float HWC
                # Transpose and move to device for Inception
                batch_tensor = torch.from_numpy(processed_batch_np.transpose((0, 3, 1, 2))).to(gpu_id) # Ensure tensor is on this GPU

                with torch.no_grad():
                   features = inception(batch_tensor)
                embeddings_list.append(features.cpu().detach().numpy()) # Move embeddings to CPU

            if embeddings_list:
                local_embeddings = np.concatenate(embeddings_list, axis=0)
            else:
                local_embeddings = np.array([]) # Handle case where 0 images were processed

            execution_logger.debug(f"Worker {worker_id} computed {len(local_embeddings)} embeddings.")

            # Store results: list of tuples (original_index, embedding)
            # Ensure embeddings are 1-to-1 with prompts_chunk and original_indices_chunk
            results_list.extend([(original_indices_chunk[j], local_embeddings[j]) for j in range(len(local_embeddings))])

        else:
             execution_logger.warning(f"Worker {worker_id} generated 0 images, skipping embedding computation.")


    except Exception as e:
        execution_logger.error(f"Worker {worker_id} on GPU {gpu_id} failed: {e}", exc_info=True)
        # Optional: Store error information in results_list or a separate error list
        # results_list.append({'error': str(e), 'worker_id': worker_id})



class hfpipe_embedding(Embedding):
    """Compute the Sentence Transformers embedding of text."""

    def __init__(self, pipeline, model_name, batch_size=4, sample_config = {}):
        """Constructor.

        :param model_name: The name of model to use
        :type model_name: str
        :param batch_size: The batch size to use for computing the embedding, defaults to 2000
        :type batch_size: int, optional
        """
        super().__init__()
        self._pipeline = pipeline
        self._model_name = model_name
        self._sample_config = sample_config
        self._batch_size = batch_size


    @property
    def column_name(self):
        """The column name to be used in the data frame."""
        return f"{EMBEDDING_COLUMN_NAME}.{type(self).__name__}.{self._model_name}"

    def compute_embedding(self, data):
        """Compute the image embedding using multiple GPUs via threading.

        :param data: The data object containing the text prompts in TEXT_DATA_COLUMN_NAME.
        :type data: :py:class:`pe.data.Data`
        :return: The data object with the computed embedding column added.
        :rtype: :py:class:`pe.data.Data`
        """
        # Filter rows that don't have this embedding computed yet
        uncomputed_data = self.filter_uncomputed_rows(data)
        if len(uncomputed_data.data_frame) == 0:
            execution_logger.info(f"Embedding: {self.column_name} already computed for all samples.")
            return data

        total_samples_to_compute = len(uncomputed_data.data_frame)
        execution_logger.info(
            f"Embedding: computing {self.column_name} for {total_samples_to_compute}/{len(data.data_frame)}"
            " samples using multiple GPUs via threading."
        )

        # Get prompts and their original indices from the uncomputed data frame
        samples_full_list = uncomputed_data.data_frame[TEXT_DATA_COLUMN_NAME].tolist()
        original_indices_full_list = uncomputed_data.data_frame.index.tolist() # Keep original indices

        # Do sample filter (regex) - apply to the full list, keeping track of indices
        pattern = r'([^:]*:|^)(.*?)(?=\.$|$)'
        samples_filtered = []
        original_indices_filtered = [] # Keep corresponding indices for filtered samples

        for i, text in enumerate(samples_full_list):
             match = re.search(pattern, str(text), re.DOTALL)
             filtered_prompt = match.group(2).strip() if match and match.group(2) else ""
             samples_filtered.append(filtered_prompt)
             original_indices_filtered.append(original_indices_full_list[i])


        total_filtered_samples = len(samples_filtered)
        if total_filtered_samples == 0:
             execution_logger.info("No samples remaining after filtering. Returning original data.")
             # Ensure column exists with NaNs if it was filtered out entirely
             if self.column_name not in data.data_frame.columns:
                  data.data_frame[self.column_name] = pd.Series([np.nan]*len(data.data_frame), index=data.data_frame.index, dtype=object)
             return data


        # --- Multi-GPU Parallel Processing with Threading ---
        available_gpus = [i for i in range(torch.cuda.device_count())]
        if not available_gpus:
             execution_logger.warning("No GPUs available. Falling back to single CPU processing (will be slow).")
             # Fallback to CPU if no GPUs
             num_gpus = 1
             target_devices = ["cpu"]
        else:
             num_gpus = len(available_gpus)
             target_devices = available_gpus
             execution_logger.info(f"Found {num_gpus} GPUs: {available_gpus}. Distributing workload.")


        # Split the filtered samples and indices into chunks for each GPU
        chunks_prompts = np.array_split(samples_filtered, num_gpus)
        chunks_indices = np.array_split(original_indices_filtered, num_gpus)

        threads = []
        all_results = [] # List to collect results from all threads (list of (index, embedding) tuples)

        # Create and start threads
        for i, gpu_id in enumerate(target_devices):
            if len(chunks_prompts[i]) == 0:
                 execution_logger.debug(f"Skipping worker {i} on device {gpu_id} as it has no prompts.")
                 continue # Skip if chunk is empty

            execution_logger.debug(f"Starting worker {i} on device {gpu_id} with {len(chunks_prompts[i])} prompts.")

            # Pass necessary objects to the worker. Be mindful of passing large mutable objects.
            # Models are moved to device within the worker. Sample config is small.
            # resize_pre and resizer are assumed to be lightweight or thread-safe.
            thread = threading.Thread(
                target=generate_and_embed_worker,
                args=(
                    gpu_id, # Pass the device identifier (int for GPU, "cpu" string)
                    chunks_prompts[i].tolist(), # Pass as list
                    chunks_indices[i].tolist(), # Pass as list
                    self._model_name,
                    self._pipeline,
                    self._sample_config,
                    self._batch_size,
                    all_results, # This list is shared, results will be appended here
                    i # worker_id for logging
                )
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        execution_logger.info(f"Waiting for {len(threads)} workers to finish...")
        for thread in tqdm(threads, desc="Overall Multi-GPU Processing"):
            thread.join()
        execution_logger.info("All workers finished.")

        # --- Process and Merge Results ---
        # all_results is a list of (original_index, embedding) tuples gathered from all threads
        if not all_results:
            execution_logger.warning("No results collected from any worker.")
            # Add the column with NaNs if no embeddings were computed
            uncomputed_data.data_frame[self.column_name] = pd.Series([np.nan]*len(uncomputed_data.data_frame), index=uncomputed_data.data_frame.index, dtype=object)
            return self.merge_computed_rows(data, uncomputed_data)


        # Sort results by original index to match the uncomputed_data.data_frame order
        # Although Pandas Series assignment by index handles order, sorting ensures predictability
        all_results.sort(key=lambda x: uncomputed_data.data_frame.index.get_loc(x[0])) # Sort by the position of the index

        # Separate indices and embeddings
        sorted_indices = [item[0] for item in all_results]
        sorted_embeddings = [item[1] for item in all_results] # This is a list of numpy arrays


        # Ensure the number of results matches the number of filtered samples
        if len(sorted_embeddings) != total_filtered_samples:
             execution_logger.warning(f"Mismatch: Expected {total_filtered_samples} embeddings, but got {len(sorted_embeddings)}. This might indicate an issue in workers (e.g., errors, skipped samples).")
             # Handle this case: potentially create a Series with NaNs for missing indices, or raise error


        # Create a pandas Series with the collected embeddings, using the original indices
        # Pandas will align based on the index
        # pd.Series expects a list of list/array-like or Series of objects if elements are variable size arrays.
        embeddings_series = pd.Series(sorted_embeddings, index=sorted_indices)

        # Assign the Series to the column in the uncomputed data frame.
        # This will correctly align based on index.
        uncomputed_data.data_frame[self.column_name] = embeddings_series


        execution_logger.info(
            f"Embedding: finished computing {self.column_name} for "
            f"{len(embeddings_series)}/{total_samples_to_compute} samples that were processed." # Log processed vs total uncomputed
        )

        # Merge the updated uncomputed rows back into the original data frame
        return self.merge_computed_rows(data, uncomputed_data)


class dpldm_embedding(Embedding):
    """Compute the Sentence Transformers embedding of text."""

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
        self._inception = InceptionV3W(path=self._temp_folder, download=True, resize_inside=False).to("cuda")
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