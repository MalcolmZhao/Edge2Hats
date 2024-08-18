# Edge2Hats: Transforming Line Drawings into Realistic Hat Images

Welcome to the Edge2Hats GitHub repository! This project leverages Conditional Generative Adversarial Networks (cGANs) to transform simple line drawings into color-filled hat images. Edge2Hats enhances the fashion design process by enabling rapid customization and bridging the gap from concept to final visualization.

## Repository Structure

- **raw_data_extraction/**
  - Contains Jupyter notebooks that utilize Selenium to extract cap images from Amazon. Various keywords were used to ensure a diverse collection of hat styles.
  
- **model_training/**
  - `edge_detection.ipynb`: A Jupyter notebook to extract hat edges from the raw images.
  - `model_training_script.py`: A Python script to run the cGAN training process using PyTorch.
  - `models/`: Contains trained model weights with different L1 penalty parameters.

- **final_product_design/**
  - `models/`: Stores the final model weights.
  - `model_edge2hats_structure.py`: Defines the generator structure for the Edge2Hats transformation.
  - `ui_edge2hat.py`: A Tkinter-powered script to generate an executable file.
  - `website_edge2hats.ipynb`: A Jupyter notebook using Gradio to build a small website interface for the model.
  
- **edge2hats_poster.pdf**: Poster detailing the project’s methodology, results, and future potential.
- **edge2hats_product_demo_video.mp4**: A demo video showcasing the product functionality.

## Dependencies

To run this project, you will need to install the following packages:

- Python 3.7 or higher
- **Jupyter Notebook**: For running and developing notebooks.
- **Selenium**: For web scraping and data extraction from Amazon.
- **PyTorch**: For building and training the cGAN model.
- **Tkinter**: For building the desktop UI.
- **Gradio**: For building the web interface.
- **OpenCV**: For image processing tasks like edge detection.
- **numpy**: For numerical computations.
- **pandas**: For data manipulation and analysis.
- **matplotlib**: For plotting and visualization.

You can install these dependencies using pip:

```bash
pip install jupyter selenium torch torchvision tkinter gradio opencv-python numpy pandas matplotlib
