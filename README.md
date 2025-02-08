## Image2Text2Image DP: Differential Privacy Data Synthesis Cross Modality
We believe that the text modality can introduce greater variation in image generation under DP constraints while not significantly compromising the quality of the generated data.

### Proposed Methodology
Following the work [Lin et al.(2024)](https://openreview.net/forum?id=YEhQs8POIo),[Xie et al.(2024)](https://arxiv.org/abs/2403.01749), we aim to exploit more potential from LLM and diffusion models, thus we proposed the following method to enhance the variety of synthesized data while preserving accuracy.

<img src="docs/images0.png" width="300">

### Experiment Setup
* Dataset: We use **LSUN dataset, bedroom class, train split** as our dataset.
* Caption: We're currently using `Salesforce/lip-image-captioning-large` as our caption llm
* PE: We use code from [DPSDA](https://github.com/microsoft/DPSDA) and follow its [example](https://github.com/microsoft/DPSDA/blob/main/example/text/pubmed_huggingface/main.py) to run Private Evolution
* Diffusion: We're currently using `stabilityai/stable-diffusion-xl-base-1.0`
* Metrics & Benchmarks: We're currently using FID and IS to evaluate the quality of synthetic data, and will use synthetic data for down-stream tasks.


### ToDo
* Examine the `textpe/run_pe.py` script to identify the factors contributing to the decline in PE data quality.
    * Try to use API from OpenAI/DeepSeek
* Enable Dense Caption in `caption/script.py`
* Use more agents for image sampling