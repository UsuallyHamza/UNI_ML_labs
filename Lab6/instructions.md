Run tests for Lab 6 (KMeans)
================================

1. Open an Anaconda Prompt or a terminal.

2. Activate the Conda environment used for these labs:

```powershell
conda activate MLlabtestenv
```

Note: If the `MLlabtestenv` environment is not already installed, follow the environment setup instructions in ../INSTRUCTIONS_for_students.md (GCR) before continuing.

3. Change directory to the lab folder that contains `lab6_test_kmeans.py`.

```powershell
cd <lab-directory-containing-test>
```

4. Run the tests for this lab with verbose output:

```bash
pytest -v lab6_test_kmeans.py
```

That's it — activate the environment, cd into the lab directory, then run `pytest -v lab6_test_kmeans.py`.
