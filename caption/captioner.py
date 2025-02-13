import base64
import io
import os
from abc import ABC, abstractmethod
from typing import List, Union, Dict
from tqdm import tqdm
from PIL import Image
import logging

from torch.utils.data import Dataset

from concurrent.futures import ThreadPoolExecutor

from transformers import pipeline

from tenacity import retry
from tenacity import retry_if_not_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential
from tenacity import before_sleep_log

import openai
from openai import BadRequestError
from openai import AuthenticationError
from openai import NotFoundError
from openai import PermissionDeniedError


class Captioner(ABC):
    @abstractmethod
    def __call__(self, images: Union[Image.Image, List[Image.Image], Dataset])-> Union[str, List[str]]:
        raise NotImplementedError("Error! Calling Captioner Abstract Class.")

class Openai_captioner(Captioner):
    """A wrapper for OpenAI LLM APIs. The following environment variables are required:

    * ``OPENAI_API_KEY``: OpenAI API key. You can get it from https://platform.openai.com/account/api-keys. Multiple
      keys can be separated by commas, and a key with the lowest current workload will be used for each request."""
    
    def __init__(self, config: Dict):
        """Constructor.

        :param config: configurations for running OpenAI API, can be observed in `config.json`
        """
        super().__init__()

        self.openai_api_key = os.environ['OPENAI_API_KEY']
        self.client = openai.OpenAI()

        self.config = config

    def __call__(self,images: Union[Image.Image, List[Image.Image], Dataset])-> Union[str, List[str]]:
        """caption the whole dataset.
        
        :param images: a single image, a list of images or an image dataset(torch)
        :type images: Union[Image.Image, List[Image.Image], Dataset]
        :return: captions of input images
        :rtype: list[str]
        """
        is_single = isinstance(images, Image.Image)
        images = [images] if is_single else images
        captions = []
        batch_size = self.config["batch_size"]

        for batch_idx in tqdm(range((len(images)+batch_size-1)//batch_size)):
            imgs = [images[idx] for idx in range(batch_idx*batch_size,(batch_idx+1)*batch_size)]
            encoded_images = [self.encode_image_from_pil(img) for img in imgs]

            with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
                responses = list(
                    tqdm(
                        executor.map(self._get_response_for_one_request, encoded_images),
                        total=len(encoded_images),
                        disable=not self.config["progress_bar"],
                    )
                )
            
            captions.extend(responses)

        return captions


    def encode_image_from_pil(self, image: Image.Image) -> str:
        """Transform PIL.Image object to Base64-encoded strings
        
        :param image: The image that needs encoding
        :type image: PIL.Image
        :return: encoded string of the image
        :rtype: str
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")  
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    

    @retry(
        retry=retry_if_not_exception_type(
            (
                BadRequestError,
                AuthenticationError,
                NotFoundError,
                PermissionDeniedError,
            )
        ),
        wait=wait_random_exponential(min=8, max=500),
        stop=stop_after_attempt(30),
        # before_sleep=before_sleep_log(execution_logger, logging.DEBUG),
    )
    def _get_response_for_one_request(self, encoded_image):
        """Get response for one caption request

        :param encoded_image: encoded image for response
        :type encoded_image: str
        :return: response for one request
        :rtype: str
        """
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                        },
                    ],
                }
            ],
            **self.config["openai_run"]
        )
        return response.choices[0].message.content



class Huggingface_captioner(Captioner):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.pipe = pipeline("image-to-text",**self.config["hf_model"])

    def __call__(self,images: Union[Image.Image, List[Image.Image], Dataset])-> Union[str, List[str]]:
        is_single = isinstance(images, Image.Image)
        images = [images] if is_single else images

        captions = []
        for caption in tqdm(self.pipe(images,**self.config['hf_run']),total=len(images)):
            captions.append(caption[0]['generated_text'])

        return captions
        