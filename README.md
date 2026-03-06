# Topic Modelling (Gensim + LDA)

Python application for topic modelling using **Gensim** and **Latent Dirichlet Allocation (LDA)**. 

## Setup

1. **Virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   
   pip install -r requirements.txt
   ```
2. **Execution:**

   ```bash
   python main.py
   ```

   Outputs are written to the `output/` directory: interactive pyLDAvis HTML, bar chart, heatmap, topic–word network, and a Plotly dashboard.
