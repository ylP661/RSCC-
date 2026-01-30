LEVIR-CC (Caption-only) data folder template
==========================================

Put the following files here (same format as original tokenizer):
- train.txt / val.txt / test.txt : each line is an image filename, e.g. train_000001.png
- vocab.json : vocabulary mapping token -> id
- tokens/ folder: each <image_id>.txt contains a JSON list of token lists

Example:
tokens/train_000001.txt
[
  ["<START>", "a", "new", "building", "appears", "<END>"],
  ["<START>", "buildings", "increased", "<END>"]
]

The trainer expects data_folder like:
<DATA_FOLDER>/train/A/*.png, <DATA_FOLDER>/train/B/*.png, etc.
