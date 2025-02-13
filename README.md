## Image2Text2Image DP: Differential Privacy Data Synthesis Cross Modality
We believe that the text modality can introduce greater variation in image generation under DP constraints while not significantly compromising the quality of the generated data. I personally name our proposed method as **Image2Text2Image DP**, or **DPITI** for short.

### Proposed Methodology
Following the work [Lin et al.(2024)](https://openreview.net/forum?id=YEhQs8POIo),[Xie et al.(2024)](https://arxiv.org/abs/2403.01749), we aim to exploit more potential from LLM and diffusion models, thus we proposed the following method to enhance the variety of synthesized data while preserving accuracy.

<img src="docs/images0.png" width="300">

### Experiment Setup
* **Dataset** : We use **LSUN dataset, bedroom class, train split** as our dataset.
* **Caption** : We're currently using `Salesforce/blip-image-captioning-large` as our caption llm
* **PE** : We use code from [DPSDA](https://github.com/microsoft/DPSDA) and follow its [example](https://github.com/microsoft/DPSDA/blob/main/example/text/pubmed_huggingface/main.py) to run Private Evolution
* **Diffusion** : We're currently using `stabilityai/stable-diffusion-xl-base-1.0`
* **Metrics & Benchmarks** : We're currently using FID and IS to evaluate the quality of synthetic data, and will use synthetic data for down-stream tasks.

### Environment
```bash
# Please prepare torch, transformers and diffusers yourself

# installing PE...
conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0
pip install "private-evolution @ git+https://github.com/microsoft/DPSDA.git"
pip install "private-evolution[image,text] @ git+https://github.com/microsoft/DPSDA.git"

# Others...
```

### Experiment Results
We post some experiment results in `doc/examples` directory.

### To-Do
* Add a logger.
* Currently we caption images with normal length.
   * Enable Dense Caption in `caption/script.py`. 
   * use OpenAI API key (gpt-4o) or 千问 API keys to generate dense caption.
* (Temporally solved) We find running PE decline text quality heavily.
   * Examine the `textpe/run_pe.py` script to identify the factors contributing to the decline in PE data quality.
   * Try to use API from OpenAI/DeepSeek.
   * `textpe/random_api_prompt.json` and `textpe/variation_api_prompt.json` has given **short** prompt to generate short enough text. Need to delete before final version.
* Currently we do sampling using a `diffusers` pipeline.
   * find a way to increase `max_token_num` for diffusion models.
   * Use more agents to do image sampling. Read more papers to see if this strategy greatly improves the quality.
   * DALL-E / Stable-Diffusion
* Compute fid on different checkpoints.
* Use caption to generate image samples directly -> serve as upper bound.
   * use huggingface llm caption
   * use OpenAI API caption