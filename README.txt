ADEPT: Ambulance Dispatch Efficiency via Policy Training

This is the repository for ADEPT, a reinforcement learning (RL) project to optimize ambulance dispatch in urban settings like New York City (NYC). It uses Proximal Policy Optimization (PPO) with KMeans clustering, tested on synthetic NYC-inspired data. This README.txt provides all you need to clone, install, and run the project, plus context and notes for future use (e.g., 5-6 years from April 2025).

---

Repository Contents
- project2.py: Main script for PPO training/evaluation (10,000 episodes default).
- cluster.py: KMeans clustering for incident zones.
- synthetic_data_generation.py: Creates synthetic EMS data mimicking NYC Open Data Portal.
- report.pdf: Semester project report (non-IEEE).
- models/ppo_agent_final_actor.keras & models/ppo_agent_final_critic.keras: PPO models trained for 10,000 episodes.
- README.txt: This file.

---

Cloning the Repository
1. Install Git:
   - Download from https://git-scm.com/downloads.
   - Verify: Open a command prompt and run "git --version" (e.g., git version 2.43.0).
2. Clone the Repo:
   - Open a command prompt or terminal.
   - Run: git clone https://github.com/workVipulPareek/ADEPT.git
   - Navigate: cd ADEPT
   - Replace workVipulPareek with your actual GitHub username.
3. Check Files:
   - Ensure all listed files are present: dir (Windows) or ls (Linux/Mac).

---

Installing Dependencies
Requirements:
- Python 3.8-3.11 (TensorFlow 2.15 compatible as of 2025).
- pip (included with Python).
- Optional: NVIDIA GPU with CUDA for faster training.

Steps:
1. Create a Virtual Environment:
   - Run: python -m venv adept_env
   - Activate (Windows): adept_env\Scripts\activate
   - Activate (Linux/Mac): source adept_env/bin/activate
   - You’ll see (adept_env) in your prompt.
2. Install Libraries:
   - Update pip: pip install --upgrade pip
   - Create a file named requirements.txt with:
     tensorflow==2.15.0
     numpy==1.26.4
     matplotlib==3.8.3
     tqdm==4.66.2
     scikit-learn==1.4.1
   - Run: pip install -r requirements.txt
3. Verify:
   - Run: python -c "import tensorflow as tf; print(tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
   - Expected: 2.15.0 and GPU details if available (e.g., [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]).

---

Running the Project
Basic Run (CPU):
1. Activate environment: adept_env\Scripts\activate
2. Run: python project2.py
   - Trains PPO for 10,000 episodes, evaluates vs. baselines, saves models/plots.

GPU Run (Windows):
1. Install CUDA/cuDNN:
   - For TensorFlow 2.15: CUDA 12.2, cuDNN 8.9 (check https://www.tensorflow.org/install/gpu).
   - Download CUDA: https://developer.nvidia.com/cuda-downloads
   - Download cuDNN: https://developer.nvidia.com/cudnn (register required).
   - Install CUDA, copy cuDNN files to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\.
2. Set PATH:
   - Add C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin to system PATH.
   - Verify: nvidia-smi (shows GPU info, e.g., RTX 3060).
3. Install GPU TensorFlow:
   - Run: pip install tensorflow[and-cuda]==2.15.0
4. Run: python project2.py
   - TensorFlow auto-uses GPU if detected.

Outputs:
- Models: models/ (e.g., ppo_agent_final_actor.keras).
- Plots: graphs/ (e.g., training_reward.png).
- Logs: logs/ (view with: tensorboard --logdir logs/, then open http://localhost:6006).

---

Theoretical Background
ADEPT optimizes ambulance dispatch:
- Problem: Assign ambulances to incidents to minimize wait times and maximize coverage.
- Approach:
  - Synthetic Data: Generated by synthetic_data_generation.py to mimic NYC EMS patterns.
  - KMeans: Clusters incidents into 5 zones (borough-like) via cluster.py.
  - PPO: RL algorithm with MDP:
    - State: Incident counts, ambulance locations, wait times, fatigue, hour.
    - Action: Assign ambulances to clusters.
    - Reward: R = 1.5 * m_kt - 0.1 * w_kt - 0.002 * T_ki (served requests, wait time, travel time).
- Training: 10,000 episodes with curriculum (regular to chaotic environments).

---

Remarks and Additional Notes
Context:
- Built as a semester project at IIIT Guwahati (2025) under Dr. Subhasish Dhal.
- Goal: Improve EMS efficiency using RL.

Future-Proofing (2030+):
- If Python 3.11 or TensorFlow 2.15 fails, try Python 3.8-3.10 or latest TF (check https://www.tensorflow.org/install).
- Hardware: 2025 GPU (e.g., RTX 3060) runs 10k episodes in hours, 100k in days—future GPUs will be faster.
- Data: Regenerate with synthetic_data_generation.py if files (rl_environment_data.json, static_allocation.npy) are missing.

Extending to 100,000 Episodes:
- Edit project2.py: Change episodes=10000 to episodes=100000 in train_agent call.
- Run as above—takes longer (days on GPU).

Disclaimer:
- Authentic Work: By Vipul Pareek (IIIT Guwahati, 2025). Uses TensorFlow PPO and scikit-learn KMeans with custom tweaks.
- Usage: Cite this repo (https://github.com/workVipulPareek/ADEPT) if used. Don’t copy without credit—meant for learning, not plagiarism.

Troubleshooting:
- GPU Fails: Runs on CPU (slower but works).
- Errors: Check TF version, missing files, or Python compatibility.

---

Contact
- Email: vipulpareek2003@gmail.com.
- Supervisor: Dr. Subhasish Dhal, IIIT Guwahati.

---
