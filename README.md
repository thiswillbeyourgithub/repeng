# repeng - Research Fork

**This is an experimental repo where I experiment with [repeng](https://github.com/vgel/repeng).**

**It was made to organise the investigation mentioned [in this issue](https://github.com/vgel/repeng/issues/27).**

**NOTE: keep in mind that this is an experimental repo, and a WIP actively maintained. I'm doing this to keep track of what I do. Don't hesitate to reach out for any remarks! But don't expect my code to be always coherent etc**

Specifically, things I intend to do are:

- Take one model as a reference
- Benchmark the model using langtest to get its reference scores on things like MMLU
- do the following comparisons also with the instruct vs base versions
- Run the benchmark again after applying the following vector to measure how badly we crippled the LLM:
    - Apply only to some layers and see how much it impacts the benchmarks
    - with 100 samples:
        - PCA
        - pacmap
        - UMAP
        - UMAP with densmap
        - PCA + 0.3 * pacmap
        - PCA + 0.3 * UMAP
        - PCA + 0.3 * UMAP with densmap
    - Again with 1000 samples
    - Again with normalization of each directions (=rescaling to have max size of 1, or applying L1, or L2)
    - Again with all layers, only the middle half layers, only the last half
    - Keep 10 UMAP dimensions, do a kmeans with k=5, apply the repeng using as vector the 1D pca of only the points in the first cluster, do that for each clusters and see if they all have a strong effect of not
    - Create a pair of good and bad intelligence-aligned examples, see if it increases its accuracy on other similar benchmarks
    - Create a pair of good and bad answers to the MMLU, see if it increases its accuracy on other similar benchmarks




# How to replicate my setup
- git clone this repo
- cd into it
- `uv venv` then activate the venv
- Install dependencies from `research/requirements.txt` with `uv pip install -r research/requirements.txt`
- install my slightly modified repeng into the venv with `uv pip install -e .`
- Also might be needed:
    - `uv pip install -U gguf`
    - installing `umap-learn` by following [those instructions](https://pypi.org/project/umap-learn/)
    - installing `pacmap` by following [those instructions](https://pypi.org/project/pacmap/)
