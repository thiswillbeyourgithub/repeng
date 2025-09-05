# repeng - Research Fork

**This is an experimental repo where I experiment with [repeng](https://github.com/vgel/repeng). It was made to organise the questions mentioned [in this issue](https://github.com/vgel/repeng/issues/27), [pr](https://github.com/vgel/repeng/pull/55) and [pr](https://github.com/vgel/repeng/pull/65).**

# Notes
- Don't hesitate to reach out!
- This is an experimental repo, that I occasionaly push to.
- I'm also doing this to keep track of what I do.

Specifically, things I intend to do are:

# Current plan:

## 1. create a calibrating suite:
    - For a specific model initially
        - Current best choice is `qwen/qwen3-4b` because it's small and *okay*. And I have terrible hardware. If you're rich and want to send me a GPU I would make great use of it!
    - if you give pairs of "dumb/smart" then ask for the model to estimate its IQ, it's easy to parse the answer to measure which layers to target and by how much etc
        - same idea with "young/old" then ask to estimate its age.
            - *Note: I was hopeful about that one but LLMs are too stuborn and insist that they are born in like 2023 or something.*
                - Actually, By asking the LLM to imagine being a human, and asked the age of that human it works.
        - same idea with "sad/happy" then ask to estimate its [BDI](https://en.wikipedia.org/wiki/Beck_Depression_Inventory) or [PHQ-9](https://en.wikipedia.org/wiki/PHQ-9) score.
        - and so on
    - we can then answer:
        a. Is the "best layer" stable across experiments
        b. Is the "best layer"'s sensitivity (strength wise) stable across experiments
        c. Is the "best layer" about the same for different size of distilled models? (gemma models)
        d. Is the "best layer" about the same for different model families? (mistral vs gemma vs llama)
        e. What is the impact of the number of samples on the reliability of those effects?
        f. What is the impact of quantization on this effect?
        g. What is the impact of longer context on this effect? And of thinking? Does the influence get amplified, fades away or is stable?

## 2. Redo this whole experience but comparing between vector extraction methods:
    - mean (the mean value of positive samples - mean value of negative samples)
    - median (the median value of positive samples - median value of negative samples)
    - [PCA](https://scikit-learn.org/stable/modules/decomposition.html)
        - pca_diff
        - pca_center
    - [kPCA](https://scikit-learn.org/stable/modules/decomposition.html)
    - [dictionary learning](https://scikit-learn.org/stable/modules/decomposition.html)
    - [ICA](https://scikit-learn.org/stable/modules/decomposition.html)
    - [NMF](https://scikit-learn.org/stable/modules/decomposition.html)
    - [UMAP](https://umap-learn.readthedocs.io/en/latest/)
    - [UMAP with densmap](https://umap-learn.readthedocs.io/en/latest/densmap_demo.html)
    - [pacmap](https://github.com/YingfanWang/PaCMAP/)

The idea is to do a grid_search (with taguchi reduction using my other project [TaguchiGridSearchConverted](https://pypi.org/project/taguchigridsearchconverter/) and store all the data into [tensorboard](https://www.tensorflow.org/tensorboard).


<details>
<summary>Click to read older ideas</summary>

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

</details>




# How to replicate my setup
- git clone this repo
- cd into it
- `uv venv` then activate the venv
- install my slightly modified repeng into the venv with `uv pip install -e .`
- Install new dependencies from `./repeng/research/requirements.txt` with `uv pip install -r ./repeng/research/requirements.txt`
- Also might be needed:
    - installing `umap-learn` by following [those instructions](https://pypi.org/project/umap-learn/)
    - installing `pacmap` by following [those instructions](https://pypi.org/project/pacmap/)
