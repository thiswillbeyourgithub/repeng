# repeng fork

This is a fork of [repeng](https://github.com/vgel/repeng/) made to organise the investigation mentioned [in this issue](https://github.com/vgel/repeng/issues/27).

Specifically, things I intend to do are:

- Take as reference the model llama3.2 8B
- Benchmark the model using langtest to get its reference scores on things like MMLU
- Run the benchmark again after applying the following vector to measure how badly we crippled the LLM:
    - Apply only to some layers and see how much it impacts the benchmarks
    - with 100 samples:
        - PCA
        - UMAP
        - UMAP with densmap
        - PCA + 0.3 * UMAP
        - PCA + 0.3 * UMAP with densmap
    - Again with 1000 samples
    - Create a pair of good and bad answers to the MMLU, see if it increases its accuracy on other similar benchmarks
