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
from accelerate import Accelerator
import re

from DPLDM.ldm.util import instantiate_from_config
from DPLDM.ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError


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


class hfpipe_xl_embedding(Embedding):
    """Compute embeddings for images generated from text prompts using Stable Diffusion XL."""

    def __init__(self, model, batch_size=4, sample_config = None, enable_accl = True):
        """Constructor.

        :param model: The name of the Stable Diffusion XL model from Hugging Face Hub
        :type model: str
        :param batch_size: The batch size to use for inference (per device when using accelerate), defaults to 4
        :type batch_size: int, optional
        :param sample_config: Additional keyword arguments for the pipeline call (e.g., num_inference_steps, generator), defaults to {}
        :type sample_config: dict, optional
        :param enable_accl: Whether to use accelerate for multi-GPU inference, defaults to True
        :type enable_accl: bool, optional
        """
        super().__init__()
        self._model_name = model
        # Ensure sample_config is a dict even if None is passed
        self._sample_config = sample_config if sample_config is not None else {}
        self._enable_accl = enable_accl
        self._batch_size = batch_size

        # Initialize Accelerator (handles device placement and distribution later)
        self._accl = Accelerator() if self._enable_accl else None

        # Load the pipeline
        # The pipeline will be moved to the correct device(s) by accelerator.prepare()
        # or manually moved to cuda/cpu if accelerate is disabled.
        execution_logger.info(f"Loading Stable Diffusion XL pipeline: {self._model_name}")
        self._pipe = StableDiffusionXLPipeline.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16, # Common for SDXL
            variant="fp16",            # Common for SDXL
            use_safetensors=True
        )

        # Prepare the pipeline with Accelerator if enabled
        if self._enable_accl and self._accl is not None:
            # prepare() handles moving the model to the correct device(s)
            self._pipe = self._accl.prepare(self._pipe)
            execution_logger.info(f"Using Accelerate with {self._accl.num_processes} processes on device: {self._accl.device}")
        else:
            # Manual device placement if Accelerate is not enabled
            if torch.cuda.is_available():
                self._pipe = self._pipe.to("cuda")
                execution_logger.info("Using single GPU (cuda). Accelerate not enabled.")
            else:
                execution_logger.warning("Using CPU. Accelerate not enabled and CUDA not available. SDXL on CPU will be very slow.")
                self._pipe = self._pipe.to("cpu")

        # FID-related components (InceptionV3 for feature extraction)
        # These components will likely run only on the main process after gathering images
        # but we initialize them considering the potential device setup.
        self._temp_folder = tempfile.TemporaryDirectory()

        # Determine device for Inception model
        _inception_device = "cpu" # Default to CPU
        if torch.cuda.is_available():
            _inception_device = "cuda"
        if self._enable_accl and self._accl is not None:
            # Use the accelerator's device for consistency if enabled and a GPU is available
            _inception_device = self._accl.device

        execution_logger.info(f"Loading InceptionV3 model on device: {_inception_device}")
        self._inception = InceptionV3W(path=self._temp_folder.name, download=True, resize_inside=False).to(_inception_device)
        self._inception.eval() # Ensure it's in eval mode for inference

        # Resizers for images before feeding to InceptionV3
        # These use PIL and numpy, less device-dependent during this stage
        self._resize_pre = make_resizer(
            library="PIL",
            quantize_after=False,
            filter="bicubic",
            output_size=(256, 256), # Target size for CleanFID preprocessing
        )
        self._resizer = build_resizer("clean") # The cleanfid specific resizer


    @property
    def column_name(self):
        """The column name to be used in the data frame."""
        # Clean up model name for column name if it contains paths or special chars
        clean_model_name = self._model_name.replace("/", "_").replace("-", "_").lower()
        return f"{EMBEDDING_COLUMN_NAME}.{type(self).__name__}.{clean_model_name}"


    def compute_embedding(self, data):
        """Compute the image embedding by generating images from text and running them through InceptionV3.

        :param data: The data object containing the text prompts in TEXT_DATA_COLUMN_NAME.
        :type data: :py:class:`pe.data.Data`
        :return: The data object with the computed embedding column added.
        :rtype: :py:class:`pe.data.Data` or None on non-main processes when accelerate is enabled.
        """
        # Filter rows that don't have this embedding computed yet
        uncomputed_data = self.filter_uncomputed_rows(data)
        if len(uncomputed_data.data_frame) == 0:
            # Use accelerate's is_main_process to avoid duplicate logs
            if self._enable_accl and self._accl is not None:
                if self._accl.is_main_process:
                    execution_logger.info(f"Embedding: {self.column_name} already computed for all samples.")
                # Non-main processes just return the original data or None, main process handles merge later
                return data # Returning data here is safer if caller expects it
            else:
                execution_logger.info(f"Embedding: {self.column_name} already computed for all samples.")
                return data


        total_samples_to_compute = len(uncomputed_data.data_frame)
        execution_logger.info(
            f"Embedding: computing {self.column_name} for {total_samples_to_compute}/{len(data.data_frame)}"
            " samples"
        )

        # Get the list of prompts for uncomputed samples.
        # Ensure this list is available on all processes initially if using manual splitting.
        samples_full_list = uncomputed_data.data_frame[TEXT_DATA_COLUMN_NAME].tolist()
        # Keep original index handy for sorting results later
        original_indices = list(uncomputed_data.data_frame.index)

        # Do sample filter (regex) - apply to the full list before distributing
        pattern = r'([^:]*:|^)(.*?)(?=\.$|$)'
        # Apply filter and keep track of original index
        indexed_samples_filtered = []
        for i, text in enumerate(samples_full_list):
            match = re.search(pattern, str(text), re.DOTALL)
            if match: # Ensure match is found
                filtered_prompt = match.group(2).strip()
                # Store original index *within the uncomputed_data frame*
                indexed_samples_filtered.append((original_indices[i], filtered_prompt))
            else:
                # Handle cases where regex doesn't match? Or assume it always matches?
                # Assuming it always matches for now, or filtered_prompt is None/empty if no match.
                # If no match, the prompt might be empty, which is still a valid generation input.
                # If you want to exclude these, filter them out here.
                # Let's keep it and pass empty strings if regex fails to match group 2.
                # Revising to handle potential None match.
                filtered_prompt = match.group(2).strip() if match and match.group(2) else ""
                indexed_samples_filtered.append((original_indices[i], filtered_prompt))


        # List to hold generated PIL images associated with their original index
        # This will be populated locally on each process, then gathered to rank 0
        local_results = [] # List of (original_index, pil_image) tuples

        if self._enable_accl and self._accl is not None:
            # --- Accelerate Multi-GPU Inference ---
            total_filtered_samples = len(indexed_samples_filtered)
            num_processes = self._accl.num_processes
            process_index = self._accl.process_index

            # Split data contiguously across processes for simpler gathering/sorting
            samples_per_process = (total_filtered_samples + num_processes - 1) // num_processes
            start_idx_global = process_index * samples_per_process
            end_idx_global = min(start_idx_global + samples_per_process, total_filtered_samples)

            # Get the slice of indexed prompts for the current process
            local_indexed_samples = indexed_samples_filtered[start_idx_global:end_idx_global]

            # Show progress bar only on the local main process (usually rank 0 within its machine)
            pbar_gen = tqdm(range(0, len(local_indexed_samples), self._batch_size),
                            desc=f"Generating (Rank {process_index}/{num_processes})",
                            disable=not self._accl.is_local_main_process or len(local_indexed_samples) == 0)

            # Generate images in batches on the current process
            for i in pbar_gen:
                 batch_indexed = local_indexed_samples[i:i+self._batch_size]
                 batch_original_indices = [item[0] for item in batch_indexed]
                 batch_prompts = [item[1] for item in batch_indexed]

                 # Ensure no gradients are computed during inference
                 with torch.no_grad():
                     # pipe() returns a list of PIL Images
                     images_batch_pil = self._pipe(batch_prompts, **self._sample_config).images

                 # Associate generated images with their original indices
                 batch_results = [(batch_original_indices[j], images_batch_pil[j]) for j in range(len(images_batch_pil))]
                 local_results.extend(batch_results)

            # Gather results from all processes to the main process (rank 0)
            # gathered_results_list_of_lists will be a list of lists of (index, pil_image) tuples.
            # It will only be populated on rank 0.
            gathered_results_list_of_lists = self._accl.gather(local_results)

            # Process results and compute embeddings only on the main process
            if self._accl.is_main_process:
                execution_logger.info("Gathering results on main process.")
                # Flatten the list of lists of results
                all_results_flat = [item for sublist in gathered_results_list_of_lists for item in sublist]

                # Sort results by original index to match the DataFrame order
                all_results_flat.sort(key=lambda x: x[0])

                # Separate images and original indices in sorted order
                sorted_images_pil = [item[1] for item in all_results_flat]
                sorted_original_indices = [item[0] for item in all_results_flat]

                execution_logger.info(f"Gathered and sorted {len(sorted_images_pil)} images on main process.")

                # Now proceed to compute embeddings using the sorted images
                images_for_inception = sorted_images_pil # Use the sorted list of PIL images

            else:
                # Non-main processes don't compute embeddings or update the DataFrame
                images_for_inception = [] # Empty list for non-main processes
                sorted_original_indices = [] # Empty list

        else:
            # --- Single-GPU / CPU Inference (No Accelerate) ---
            execution_logger.info("Running inference on a single device.")

            # Extract prompts and indices for single-device processing
            single_device_indexed_samples = indexed_samples_filtered
            sorted_original_indices = [item[0] for item in single_device_indexed_samples]
            single_device_prompts = [item[1] for item in single_device_indexed_samples]


            single_device_generated_images_pil = []
            pbar_gen = tqdm(range(0, len(single_device_prompts), self._batch_size),
                            desc="Generating (Single Device)")

            # Generate images in batches
            for i in pbar_gen:
                batch_prompts = single_device_prompts[i:i+self._batch_size]

                with torch.no_grad():
                    images_batch_pil = self._pipe(batch_prompts, **self._sample_config).images
                single_device_generated_images_pil.extend(images_batch_pil)

            images_for_inception = single_device_generated_images_pil # This is already in the correct order


        # --- Compute Inception Embeddings (Happens only on Main Process if accelerate is enabled) ---
        embeddings = None # Initialize embeddings

        # Check if we are on the main process (or if accelerate is disabled) before computing embeddings
        if not self._enable_accl or (self._accl is not None and self._accl.is_main_process):

            if not images_for_inception:
                execution_logger.warning("No images available to compute embeddings after generation.")
                # If no images were generated, return an empty embeddings array
                embeddings = np.array([])
            else:
                execution_logger.info(f"Computing Inception embeddings for {len(images_for_inception)} images.")

                # Convert list of PIL images to a NumPy array (N, H, W, C)
                # SDXL generates fixed size images, so stacking is straightforward
                images_np = np.stack([np.array(img) for img in images_for_inception], axis=0)

                # If images are grayscale, repeat channels (Inception expects 3 channels)
                if images_np.shape[-1] == 1:
                    images_np = np.repeat(images_np, 3, axis=-1) # Use axis=-1 for the last dimension

                embeddings_list = []
                # Use the same batch size for Inception processing or define a separate one
                inception_batch_size = self._batch_size

                pbar_inception = tqdm(range(0, len(images_np), inception_batch_size),
                                      desc="Computing Inception Embeddings",
                                      # Show progress bar only on main process
                                      disable=False if not self._enable_accl else not self._accl.is_main_process)

                # Process images in batches for InceptionV3
                for i in pbar_inception:
                    batch_np = images_np[i : i + inception_batch_size] # NumPy batch (N, H, W, C)

                    # Apply CleanFID preprocessing steps batch-wise (convert to uint8, then resize)
                    # The cleanfid `build_resizer("clean")` expects numpy HWC uint8
                    processed_batch_np = []
                    for img_np in batch_np:
                        # img_np is (H, W, C)
                        img_uint8 = to_uint8(img_np, min=0, max=255) # Ensure this matches image range
                        img_resized_np = self._resizer(img_uint8) # Apply cleanfid resizer (returns numpy HWC)
                        processed_batch_np.append(img_resized_np)

                    # Stack processed numpy images and convert to torch tensor (N, C, H, W)
                    # Inception model expects NCHW format
                    processed_batch_np = np.stack(processed_batch_np, axis=0).transpose((0, 3, 1, 2))
                    batch_tensor = torch.from_numpy(processed_batch_np).to(self._inception.device) # Move tensor to inception device

                    # Get inception features
                    with torch.no_grad(): # Ensure no gradients for Inception
                       features = self._inception(batch_tensor)
                    # Move features back to CPU and convert to numpy
                    embeddings_list.append(features.cpu().detach().numpy())

                # Concatenate all batch embeddings into a single numpy array
                if embeddings_list:
                    embeddings = np.concatenate(embeddings_list, axis=0)
                else:
                    embeddings = np.array([]) # Handle case where loops didn't run (e.g. 0 images)


            # --- Update DataFrame (Happens only on Main Process if accelerate is enabled) ---
            # The 'embeddings' array is now in the same order as 'sorted_original_indices'
            # We need to assign these embeddings back to the original DataFrame indices in uncomputed_data.
            # pd.Series handles alignment by index automatically if we pass the original index.
            if len(embeddings) > 0:
                 # Create a pandas Series with original indices
                 embeddings_series = pd.Series(list(embeddings), index=sorted_original_indices)
                 # Assign the Series to the column in the uncomputed data frame.
                 # This will correctly align based on index.
                 uncomputed_data.data_frame[self.column_name] = embeddings_series
            else:
                 # If no embeddings were computed (0 images), add an empty column or fill with NaNs/zeros?
                 # Depends on desired behavior. Let's add an empty column for the relevant indices.
                 uncomputed_data.data_frame[self.column_name] = pd.Series([np.nan]*len(uncomputed_data.data_frame), index=uncomputed_data.data_frame.index, dtype=object)


            execution_logger.info(
                f"Embedding: finished computing {self.column_name} for "
                f"{total_samples_to_compute}/{len(data.data_frame)} samples"
            )

            # Merge the updated uncomputed rows back into the original data frame
            final_data = self.merge_computed_rows(data, uncomputed_data)

            # Return the final data frame on the main process
            return final_data

        else:
            # --- Non-main processes in Accelerate mode ---
            # These processes finish after gathering results and wait implicitly
            # or return the original data structure. The main process's return value is used.
            execution_logger.debug(f"Rank {process_index} finished local processing and gathering.")
            return data # Returning the original data here is common practice
        


class hfpipe_embedding(Embedding):
    """Compute embeddings for images generated from text prompts using Stable Diffusion XL."""

    def __init__(self, model, batch_size=4, sample_config = None, enable_accl = True):
        """Constructor.

        :param model: The name of the Stable Diffusion XL model from Hugging Face Hub
        :type model: str
        :param batch_size: The batch size to use for inference (per device when using accelerate), defaults to 4
        :type batch_size: int, optional
        :param sample_config: Additional keyword arguments for the pipeline call (e.g., num_inference_steps, generator), defaults to {}
        :type sample_config: dict, optional
        :param enable_accl: Whether to use accelerate for multi-GPU inference, defaults to True
        :type enable_accl: bool, optional
        """
        super().__init__()
        self._model_name = model
        # Ensure sample_config is a dict even if None is passed
        self._sample_config = sample_config if sample_config is not None else {}
        self._enable_accl = enable_accl
        self._batch_size = batch_size

        # Initialize Accelerator (handles device placement and distribution later)
        self._accl = Accelerator() if self._enable_accl else None

        # Load the pipeline
        # The pipeline will be moved to the correct device(s) by accelerator.prepare()
        # or manually moved to cuda/cpu if accelerate is disabled.
        execution_logger.info(f"Loading Stable Diffusion XL pipeline: {self._model_name}")
        self._pipe = DiffusionPipeline.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16, # Common for SDXL
            variant="fp16",            # Common for SDXL
            use_safetensors=True
        )

        # Prepare the pipeline with Accelerator if enabled
        if self._enable_accl and self._accl is not None:
            # prepare() handles moving the model to the correct device(s)
            self._pipe = self._accl.prepare(self._pipe)
            execution_logger.info(f"Using Accelerate with {self._accl.num_processes} processes on device: {self._accl.device}")
        else:
            # Manual device placement if Accelerate is not enabled
            if torch.cuda.is_available():
                self._pipe = self._pipe.to("cuda")
                execution_logger.info("Using single GPU (cuda). Accelerate not enabled.")
            else:
                execution_logger.warning("Using CPU. Accelerate not enabled and CUDA not available. SDXL on CPU will be very slow.")
                self._pipe = self._pipe.to("cpu")

        # FID-related components (InceptionV3 for feature extraction)
        # These components will likely run only on the main process after gathering images
        # but we initialize them considering the potential device setup.
        self._temp_folder = tempfile.TemporaryDirectory()

        # Determine device for Inception model
        _inception_device = "cpu" # Default to CPU
        if torch.cuda.is_available():
            _inception_device = "cuda"
        if self._enable_accl and self._accl is not None:
            # Use the accelerator's device for consistency if enabled and a GPU is available
            _inception_device = self._accl.device

        execution_logger.info(f"Loading InceptionV3 model on device: {_inception_device}")
        self._inception = InceptionV3W(path=self._temp_folder.name, download=True, resize_inside=False).to(_inception_device)
        self._inception.eval() # Ensure it's in eval mode for inference

        # Resizers for images before feeding to InceptionV3
        # These use PIL and numpy, less device-dependent during this stage
        self._resize_pre = make_resizer(
            library="PIL",
            quantize_after=False,
            filter="bicubic",
            output_size=(256, 256), # Target size for CleanFID preprocessing
        )
        self._resizer = build_resizer("clean") # The cleanfid specific resizer


    @property
    def column_name(self):
        """The column name to be used in the data frame."""
        # Clean up model name for column name if it contains paths or special chars
        clean_model_name = self._model_name.replace("/", "_").replace("-", "_").lower()
        return f"{EMBEDDING_COLUMN_NAME}.{type(self).__name__}.{clean_model_name}"


    def compute_embedding(self, data):
        """Compute the image embedding by generating images from text and running them through InceptionV3.

        :param data: The data object containing the text prompts in TEXT_DATA_COLUMN_NAME.
        :type data: :py:class:`pe.data.Data`
        :return: The data object with the computed embedding column added.
        :rtype: :py:class:`pe.data.Data` or None on non-main processes when accelerate is enabled.
        """
        # Filter rows that don't have this embedding computed yet
        uncomputed_data = self.filter_uncomputed_rows(data)
        if len(uncomputed_data.data_frame) == 0:
            # Use accelerate's is_main_process to avoid duplicate logs
            if self._enable_accl and self._accl is not None:
                if self._accl.is_main_process:
                    execution_logger.info(f"Embedding: {self.column_name} already computed for all samples.")
                # Non-main processes just return the original data or None, main process handles merge later
                return data # Returning data here is safer if caller expects it
            else:
                execution_logger.info(f"Embedding: {self.column_name} already computed for all samples.")
                return data


        total_samples_to_compute = len(uncomputed_data.data_frame)
        execution_logger.info(
            f"Embedding: computing {self.column_name} for {total_samples_to_compute}/{len(data.data_frame)}"
            " samples"
        )

        # Get the list of prompts for uncomputed samples.
        # Ensure this list is available on all processes initially if using manual splitting.
        samples_full_list = uncomputed_data.data_frame[TEXT_DATA_COLUMN_NAME].tolist()
        # Keep original index handy for sorting results later
        original_indices = list(uncomputed_data.data_frame.index)

        # Do sample filter (regex) - apply to the full list before distributing
        pattern = r'([^:]*:|^)(.*?)(?=\.$|$)'
        # Apply filter and keep track of original index
        indexed_samples_filtered = []
        for i, text in enumerate(samples_full_list):
             match = re.search(pattern, str(text), re.DOTALL)
             if match: # Ensure match is found
                 filtered_prompt = match.group(2).strip()
                 # Store original index *within the uncomputed_data frame*
                 indexed_samples_filtered.append((original_indices[i], filtered_prompt))
             else:
                 # Handle cases where regex doesn't match? Or assume it always matches?
                 # Assuming it always matches for now, or filtered_prompt is None/empty if no match.
                 # If no match, the prompt might be empty, which is still a valid generation input.
                 # If you want to exclude these, filter them out here.
                 # Let's keep it and pass empty strings if regex fails to match group 2.
                 # Revising to handle potential None match.
                 filtered_prompt = match.group(2).strip() if match and match.group(2) else ""
                 indexed_samples_filtered.append((original_indices[i], filtered_prompt))


        # List to hold generated PIL images associated with their original index
        # This will be populated locally on each process, then gathered to rank 0
        local_results = [] # List of (original_index, pil_image) tuples

        if self._enable_accl and self._accl is not None:
            # --- Accelerate Multi-GPU Inference ---
            total_filtered_samples = len(indexed_samples_filtered)
            num_processes = self._accl.num_processes
            process_index = self._accl.process_index

            # Split data contiguously across processes for simpler gathering/sorting
            samples_per_process = (total_filtered_samples + num_processes - 1) // num_processes
            start_idx_global = process_index * samples_per_process
            end_idx_global = min(start_idx_global + samples_per_process, total_filtered_samples)

            # Get the slice of indexed prompts for the current process
            local_indexed_samples = indexed_samples_filtered[start_idx_global:end_idx_global]

            # Show progress bar only on the local main process (usually rank 0 within its machine)
            pbar_gen = tqdm(range(0, len(local_indexed_samples), self._batch_size),
                            desc=f"Generating (Rank {process_index}/{num_processes})",
                            disable=not self._accl.is_local_main_process or len(local_indexed_samples) == 0)

            # Generate images in batches on the current process
            for i in pbar_gen:
                 batch_indexed = local_indexed_samples[i:i+self._batch_size]
                 batch_original_indices = [item[0] for item in batch_indexed]
                 batch_prompts = [item[1] for item in batch_indexed]

                 # Ensure no gradients are computed during inference
                 with torch.no_grad():
                     # pipe() returns a list of PIL Images
                     images_batch_pil = self._pipe(batch_prompts, **self._sample_config).images

                 # Associate generated images with their original indices
                 batch_results = [(batch_original_indices[j], images_batch_pil[j]) for j in range(len(images_batch_pil))]
                 local_results.extend(batch_results)

            # Gather results from all processes to the main process (rank 0)
            # gathered_results_list_of_lists will be a list of lists of (index, pil_image) tuples.
            # It will only be populated on rank 0.
            gathered_results_list_of_lists = self._accl.gather(local_results)

            # Process results and compute embeddings only on the main process
            if self._accl.is_main_process:
                execution_logger.info("Gathering results on main process.")
                # Flatten the list of lists of results
                all_results_flat = [item for sublist in gathered_results_list_of_lists for item in sublist]

                # Sort results by original index to match the DataFrame order
                all_results_flat.sort(key=lambda x: x[0])

                # Separate images and original indices in sorted order
                sorted_images_pil = [item[1] for item in all_results_flat]
                sorted_original_indices = [item[0] for item in all_results_flat]

                execution_logger.info(f"Gathered and sorted {len(sorted_images_pil)} images on main process.")

                # Now proceed to compute embeddings using the sorted images
                images_for_inception = sorted_images_pil # Use the sorted list of PIL images

            else:
                # Non-main processes don't compute embeddings or update the DataFrame
                images_for_inception = [] # Empty list for non-main processes
                sorted_original_indices = [] # Empty list

        else:
            # --- Single-GPU / CPU Inference (No Accelerate) ---
            execution_logger.info("Running inference on a single device.")

            # Extract prompts and indices for single-device processing
            single_device_indexed_samples = indexed_samples_filtered
            sorted_original_indices = [item[0] for item in single_device_indexed_samples]
            single_device_prompts = [item[1] for item in single_device_indexed_samples]


            single_device_generated_images_pil = []
            pbar_gen = tqdm(range(0, len(single_device_prompts), self._batch_size),
                            desc="Generating (Single Device)")

            # Generate images in batches
            for i in pbar_gen:
                batch_prompts = single_device_prompts[i:i+self._batch_size]

                with torch.no_grad():
                    images_batch_pil = self._pipe(batch_prompts, **self._sample_config).images
                single_device_generated_images_pil.extend(images_batch_pil)

            images_for_inception = single_device_generated_images_pil # This is already in the correct order


        # --- Compute Inception Embeddings (Happens only on Main Process if accelerate is enabled) ---
        embeddings = None # Initialize embeddings

        # Check if we are on the main process (or if accelerate is disabled) before computing embeddings
        if not self._enable_accl or (self._accl is not None and self._accl.is_main_process):

            if not images_for_inception:
                execution_logger.warning("No images available to compute embeddings after generation.")
                # If no images were generated, return an empty embeddings array
                embeddings = np.array([])
            else:
                execution_logger.info(f"Computing Inception embeddings for {len(images_for_inception)} images.")

                # Convert list of PIL images to a NumPy array (N, H, W, C)
                # SDXL generates fixed size images, so stacking is straightforward
                images_np = np.stack([np.array(img) for img in images_for_inception], axis=0)

                # If images are grayscale, repeat channels (Inception expects 3 channels)
                if images_np.shape[-1] == 1:
                    images_np = np.repeat(images_np, 3, axis=-1) # Use axis=-1 for the last dimension

                embeddings_list = []
                # Use the same batch size for Inception processing or define a separate one
                inception_batch_size = self._batch_size

                pbar_inception = tqdm(range(0, len(images_np), inception_batch_size),
                                      desc="Computing Inception Embeddings",
                                      # Show progress bar only on main process
                                      disable=False if not self._enable_accl else not self._accl.is_main_process)

                # Process images in batches for InceptionV3
                for i in pbar_inception:
                    batch_np = images_np[i : i + inception_batch_size] # NumPy batch (N, H, W, C)

                    # Apply CleanFID preprocessing steps batch-wise (convert to uint8, then resize)
                    # The cleanfid `build_resizer("clean")` expects numpy HWC uint8
                    processed_batch_np = []
                    for img_np in batch_np:
                        # img_np is (H, W, C)
                        img_uint8 = to_uint8(img_np, min=0, max=255) # Ensure this matches image range
                        img_resized_np = self._resizer(img_uint8) # Apply cleanfid resizer (returns numpy HWC)
                        processed_batch_np.append(img_resized_np)

                    # Stack processed numpy images and convert to torch tensor (N, C, H, W)
                    # Inception model expects NCHW format
                    processed_batch_np = np.stack(processed_batch_np, axis=0).transpose((0, 3, 1, 2))
                    batch_tensor = torch.from_numpy(processed_batch_np).to(self._inception.device) # Move tensor to inception device

                    # Get inception features
                    with torch.no_grad(): # Ensure no gradients for Inception
                       features = self._inception(batch_tensor)
                    # Move features back to CPU and convert to numpy
                    embeddings_list.append(features.cpu().detach().numpy())

                # Concatenate all batch embeddings into a single numpy array
                if embeddings_list:
                    embeddings = np.concatenate(embeddings_list, axis=0)
                else:
                    embeddings = np.array([]) # Handle case where loops didn't run (e.g. 0 images)


            # --- Update DataFrame (Happens only on Main Process if accelerate is enabled) ---
            # The 'embeddings' array is now in the same order as 'sorted_original_indices'
            # We need to assign these embeddings back to the original DataFrame indices in uncomputed_data.
            # pd.Series handles alignment by index automatically if we pass the original index.
            if len(embeddings) > 0:
                 # Create a pandas Series with original indices
                 embeddings_series = pd.Series(list(embeddings), index=sorted_original_indices)
                 # Assign the Series to the column in the uncomputed data frame.
                 # This will correctly align based on index.
                 uncomputed_data.data_frame[self.column_name] = embeddings_series
            else:
                 # If no embeddings were computed (0 images), add an empty column or fill with NaNs/zeros?
                 # Depends on desired behavior. Let's add an empty column for the relevant indices.
                 uncomputed_data.data_frame[self.column_name] = pd.Series([np.nan]*len(uncomputed_data.data_frame), index=uncomputed_data.data_frame.index, dtype=object)


            execution_logger.info(
                f"Embedding: finished computing {self.column_name} for "
                f"{total_samples_to_compute}/{len(data.data_frame)} samples"
            )

            # Merge the updated uncomputed rows back into the original data frame
            final_data = self.merge_computed_rows(data, uncomputed_data)

            # Return the final data frame on the main process
            return final_data

        else:
            # --- Non-main processes in Accelerate mode ---
            # These processes finish after gathering results and wait implicitly
            # or return the original data structure. The main process's return value is used.
            execution_logger.debug(f"Rank {process_index} finished local processing and gathering.")
            return data # Returning the original data here is common practice
        


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
