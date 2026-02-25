#!/bin/bash
echo "=== Setting up pipeline ==="
python main.py --setup

echo "=== Starting Streamlit ==="
streamlit run app.py --server.address 0.0.0.0 --server.port 8501