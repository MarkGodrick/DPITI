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
from tenacity import retry_all
from tenacity import retry_if_not_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential
from tenacity import before_sleep_log

import openai
from openai import BadRequestError
from openai import AuthenticationError
from openai import NotFoundError
from openai import PermissionDeniedError

from google import genai
from google.genai import types
import google.genai.errors as errors


class Captioner(ABC):
    @abstractmethod
    def __call__(self, images: Union[Image.Image, List[Image.Image], Dataset])-> Union[str, List[str]]:
        raise NotImplementedError("Error! Calling Captioner Abstract Class.")

class Openai_captioner(Captioner):
    """A wrapper for OpenAI LLM APIs. The following environment variables are required:

    * ``OPENAI_API_KEY``: OpenAI API key. You can get it from https://platform.openai.com/account/api-keys."""
    
    def __init__(self, config: Dict):
        """Constructor.

        :param config: configurations for running OpenAI API, can be observed in `config.json`
        :type config: Dict
        :param api: name for the environment variable that stores API key.
        :type api: str
        """
        super().__init__()

        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
        dataset_len = len(images)
        captions = []
        batch_size = self.config["batch_size"]

        for batch_idx in tqdm(range((len(images)+batch_size-1)//batch_size)):
            imgs = [images[idx] for idx in range(batch_idx*batch_size,(batch_idx+1)*batch_size) if idx<dataset_len]
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
        stop=stop_after_attempt(10),
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

class Gemini_captioner(Captioner):
    """A wrapper for Gemini LLM APIs. The following environment variables are required:

    * ``GEMINI_API_KEY``: Gemini API key. You can get one from https://aistudio.google.com/app/apikey """
    
    def __init__(self, config: Dict):
        """Constructor
        
        :param config: configuration for gemini api
        :type config: dict"""
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.config = config
    
    def __call__(self,images: Union[Image.Image, List[Image.Image], Dataset]):
        """caption the whole dataset.
        
        :param images: a single image, a list of images or an image dataset(torch)
        :type images: Union[Image.Image, List[Image.Image], Dataset]
        :return: captions of input images
        :rtype: list[str]
        """
        is_single = isinstance(images, Image.Image)
        images = [images] if is_single else images
        dataset_len = len(images)
        captions = []
        batch_size = self.config["batch_size"]

        for batch_idx in tqdm(range((len(images)+batch_size-1)//batch_size)):
            imgs = [images[idx] for idx in range(batch_idx*batch_size,(batch_idx+1)*batch_size) if idx<dataset_len]

            # with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            #     responses = list(
            #         tqdm(
            #             executor.map(self._get_response_for_one_request, imgs),
            #             total=len(imgs),
            #             disable=not self.config["progress_bar"],
            #         )
            #     )
            
            # captions.extend(responses)

            for img in tqdm(imgs):
                captions.append(self._get_response_for_one_request(img))

            # captions.extend(self._get_response_for_one_batch(imgs))

        return captions
    
    @retry(
        retry=retry_if_not_exception_type((
            errors.ClientError,
            errors.FunctionInvocationError,
            errors.UnknownFunctionCallArgumentError,
            errors.UnsupportedFunctionError
        )),
        wait=wait_random_exponential(min=8, max=100),
        stop=stop_after_attempt(5),
        # before_sleep=before_sleep_log(execution_logger, logging.DEBUG),
    )
    def _get_response_for_one_batch(self, images: List[Image.Image]):
        """Getting captions for a batch.

        :param images: images that needs to be captioned.
        :type images: List[Image.Image]
        :return: captions of images
        :rtype: List[str]
        """
        chat = self.client.chats.create(
            model = self.config["model"],
            config = types.GenerateContentConfig(**self.config["gemini_run"])
        )

        responses = []
        for img in tqdm(images):
            response = chat.send_message([img,self.config['prompts']])
            responses.append(response.text)

        return responses

    @retry(
        retry=retry_if_not_exception_type(
            (
                errors.ClientError,
                errors.FunctionInvocationError,
                errors.UnknownFunctionCallArgumentError,
                errors.UnsupportedFunctionError
            )
        ),
        wait=wait_random_exponential(min=8, max=100),
        stop=stop_after_attempt(5),
        # before_sleep=before_sleep_log(execution_logger, logging.DEBUG),
    )
    def _get_response_for_one_request(self, image: Image.Image):
        """Get response for one caption request

        :param image: input image for response
        :type image: PIL.Image.Image
        :return: response for one request
        :rtype: str
        """
        response = self.client.models.generate_content(
            model = self.config["model"],
            contents=[image,self.config["prompts"]],
            config=types.GenerateContentConfig(**self.config["gemini_run"])
        )
        
        return response.text



class Qwen_captioner(Captioner):
    """A wrapper for Qwen LLM APIs. The following environment variables are required:

    * ``DASHSCOPE_API_KEY``: Qwen API key. You can get it from ."""
    
    def __init__(self, config: Dict):
        """Constructor.

        :param config: configurations for running OpenAI API, can be observed in `config.json`
        :type config: Dict
        :param api: name for the environment variable that stores API key.
        :type api: str
        """
        super().__init__()

        self.client = openai.OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

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
        dataset_len = len(images)
        captions = []
        batch_size = self.config["batch_size"]

        for batch_idx in tqdm(range((len(images)+batch_size-1)//batch_size)):
            imgs = [images[idx] for idx in range(batch_idx*batch_size,(batch_idx+1)*batch_size) if idx<dataset_len]
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
    

    # @retry(
    #     retry=retry_if_not_exception_type(
    #         (
    #             BadRequestError,
    #             AuthenticationError,
    #             NotFoundError,
    #             PermissionDeniedError,
    #         )
    #     ),
    #     wait=wait_random_exponential(min=8, max=500),
    #     stop=stop_after_attempt(10),
    #     # before_sleep=before_sleep_log(execution_logger, logging.DEBUG),
    # )
    def _get_response_for_one_request(self, encoded_image):
        """Get response for one caption request

        :param encoded_image: encoded image for response
        :type encoded_image: str
        :return: response for one request
        :rtype: str
        """
        try:
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
                **self.config["qwen_run"]
            )
            return response.choices[0].message.content
        except BadRequestError as e:
            print(f"Error occurred: {e}")
            return ""



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
        