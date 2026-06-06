🧠 NeuroScan AI — End-to-End Brain Tumor Detection with XAI

An End-to-End Deep Learning web application developed during the industrial attachment at Techknowgram Ltd. > NeuroScan AI leverages a fine-tuned ResNet-50 architecture to classify brain tumors into four categories (Glioma, Meningioma, No Tumor, Pituitary) with high diagnostic confidence. Utilizing Grad-CAM (Explainable AI), the platform visualizes neural attention hotspots in real-time, bridging the clinical trust gap.

⚠️ Medical Disclaimer

IMPORTANT CLINICAL NOTICE: This software tool is developed strictly for educational, research, and preliminary screening purposes. It does not constitute medical advice or substitute professional radiological evaluation. Under no circumstances should diagnostic determinations or clinical pathways be decided solely based on this system's output.

📌 Table of Contents

About the Project

System Architecture

Key Features

Technologies & Frameworks

Repository Directory Structure

Local Installation & Setup

Cloud Deployment (Render)

Industrial Attachment Context

License

💡 About the Project

Traditional Deep Learning models operate as black-boxes, outputting classification probabilities without reasoning. NeuroScan AI solves this bottleneck by coupling high-performance ResNet-50 classifiers with Grad-CAM (Gradient-weighted Class Activation Mapping).

When a radiologist uploads an MRI scan, the system does not just say "Glioma" — it highlights the exact anatomical slice and pixel-region that led to the prediction, instantly providing validation criteria.

⚙️ System Architecture

The application is deployed using a lightweight, highly secure Single-Origin Architecture where a fast Python server serves both the high-performance diagnostic API and the compiled frontend layout.

       [ Radiologist / User Interface (Tailwind CSS + JS) ]
                                |
             (Drag-and-Drop Single or Batch MRI Uploads)
                                |
                                v
                [ FastAPI Application Server ]
                 /                         \
    [ Single-Image Pipeline ]       [ Batch Processing Loop ]
                 \                         /
                  v                       v
               [ Core Deep Learning Engine (PyTorch) ]
               +-------------------------------------+
               |  - ResNet-50 Inference Pipeline     |
               |  - Softmax Confidence Solver        |
               |  - Grad-CAM Feature Map Generation  |
               +-------------------------------------+
                                |
                     (JSON Response Package)
                                |
                                v
               [ Diagnostic UI Rendering Engine ]
               |  - Real-Time Activation Maps Overlay|
               |  - Multi-Confidence Dynamic Tables  |
               |  - On-Demand Client PDF Generation  |


✨ Key Features

🔬 Dual Diagnostic Pipelines: - Single Scan Engine: Fast, interactive inspection of a single slice.

Multi-Image Batch Processing: Upload multiple MRI scans concurrently; the backend computes independent matrix inferences and renders dynamic multi-confidence scoring tables side-by-side.

🎯 Grad-CAM Interpretability: Targets layer4[-1] of ResNet-50 to output clean visual heatmaps.

📋 On-the-Fly Reporting: Instant PDF clinical report compiles diagnostic class, system confidence, original scan, and the Grad-CAM activation map completely client-side.

🏎️ CPU-Optimized Footprint: Customized PyTorch CPU builds ensure execution inside cloud environments with strict RAM footprints (512MB).

🛠️ Technologies & Frameworks

Deep Learning Engine: PyTorch, Torchvision, Grad-CAM (by Jacob Gildenblat)

API Backend: FastAPI, Uvicorn (ASGI)

Front-End Portal: Tailwind CSS (Modern Dark Mode UI), JavaScript (ES6 Fetch API)

Report Compiler: jsPDF Library

📂 Repository Directory Structure

brain-tumor-app/
│
├── backend/
│   ├── main.py                 # FastAPI Single-Origin Server & API Pipelines
│   ├── requirements.txt        # Production Python dependencies
│   └── resnet_best.pth         # Trained ResNet-50 weight matrix (Omitted via .gitignore)
│
├── frontend/
│   └── index.html              # Responsive Tailwind-based Diagnostic Portal
│
└── README.md                   # Technical documentation


💻 Local Installation & Setup

Follow these steps to run the complete diagnostic stack on your local machine:

1. Clone the Repository

git clone [https://github.com/Saymon72ahmed/brain-tumor-app.git](https://github.com/Saymon72ahmed/brain-tumor-app.git)
cd brain-tumor-app


2. Set Up a Virtual Environment (Recommended)

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies

pip install -r backend/requirements.txt


4. Weights Placement

Ensure your trained model weight file resnet_best.pth is placed inside the backend/ folder before launching the server.

5. Launch the FastAPI Application

python backend/main.py


The ASGI application will start, and your default web browser will automatically open to http://127.0.0.1:8000.

☁️ Cloud Deployment (Render)

This project is optimized for direct hosting on Render using a stable Python 3.11 environment.

Deployment Parameters:

Service Type: Web Service

Build Command: pip install -r backend/requirements.txt

Start Command: uvicorn backend.main:app --host 0.0.0.0 --port $PORT

Environment Variables:

PYTHON_VERSION = 3.11.9

💡 Optimization Trick: The production requirements.txt includes --index-url https://download.pytorch.org/whl/cpu which bypasses heavy GPU packages, successfully containerizing the system under Render's free CPU limits.

🤝 Industrial Attachment Context

This project was developed during an internship program at Techknowgram Ltd as part of an undergraduate industrial attachment defense curriculum.

Special thanks to:

Md. Istiak Adnan Polash, Assistant Professor & Academic Supervisor.

Engg. Md Allmamun Ridoy, Industrial Supervisor & Project Manager at Techknowgram Ltd.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
