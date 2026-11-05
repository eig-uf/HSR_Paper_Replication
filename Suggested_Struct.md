HSR-Replication/
│
├── LIBERO/                     # Existing benchmark
│
├── notebooks/
│   ├── 00_data_exploration.ipynb    # Completed
│   ├── 01_dataset.ipynb        # Completed
│   ├── 02_skill_discovery.ipynb # Completed
│   ├── 03_task_planning.ipynb
│   ├── 04_skill_retrieval.ipynb
│   ├── 05_evaluation.ipynb
|   ├── XX_MORE_AS_NEEDED
│
├── src/
│   └── hsr/
│       ├── data.py             # Dataset loading/indexing
│       ├── skills.py           # BERT + K-means
│       ├── planner.py          # LLM task decomposition
│       ├── retrieval.py        # Semantic retrieval
│       ├── behavior.py         # VAE behavior features
│       └── adaptation.py       # SmolVLA fine-tuning
│
│
├── paper_notes.md
└── README.md