# Result folder

You are in the `./results` folder, where the results of research are stored.

## File Organization

In `./results` are two folders: `reproductible_scripts/` and `tensorboard_logs/`. The scripts are appended with a commit hash (e.g. `grid_search_8f8a13d0bd468afbf6ea10e33005ec62c05e7e20.py`), this hash corresponds to a commit in the `research_setup` branch.
- To reproduce the result:
    - create a new git branch, reset to that commit, then run via `python ./repeng/research/grid_search.py`.
        - Or to avoid dealing with branches, just move the script to `./repeng/research/grid_search.py`, execute it with `python ./repeng/research/grid_search.py`
    - To open the results, use `tensorboard --logdir ./result/tensorboard_logs/some_dir`. The output will contain a link like `TensorBoard 2.20.0 at http://localhost:6006/` that you must open in a browser.

## How to read the results
