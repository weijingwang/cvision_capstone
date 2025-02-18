# Using GloVe for Word Classification in Python

This guide explains how to download and set up the GloVe word embeddings for use in our object classification project.

---

## 1. Download the GloVe File

1. **Visit the official GloVe website:**
   - [GloVe Pretrained Word Embeddings](https://nlp.stanford.edu/projects/glove/)

2. **Download the `glove.6B.zip` file**, which contains multiple versions of word embeddings.
   - Direct link: [glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip)

3. **Extract the downloaded `.zip` file.**
   - You should see files like:
     ```
     glove.6B.50d.txt
     glove.6B.100d.txt
     glove.6B.200d.txt
     glove.6B.300d.txt
     ```

---

## 2. Move the File to a Known Path

Move the extracted GloVe file (`glove.6B.100d.txt` or any other dimension version) to a **fixed location** on your system.

---

## 3. Update the Path in Your Code

Modify the `glove_file_path` in your script to match the actual location of the GloVe file:

---

## 3. Load GloVe Word Embeddings in Python
Use the following code snippet to load the GloVe embeddings into a dictionary:

import numpy as np

embeddings_dict = {}
with open(glove_file_path, "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype="float32")
        embeddings_dict[word] = vector

print("GloVe embeddings loaded successfully!")
