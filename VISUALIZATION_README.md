# BraTS Visualization for Paper Figures

This directory contains visualization scripts for creating publication-ready figures comparing baseline vs attention models for BraTS tumor segmentation.

## Installation

```bash
pip install -r requirements_visualization.txt
```

## Scripts Overview

### 1. `visualize.py` - 2D Multi-planar Visualizations

Creates overlay visualizations comparing ground truth vs predictions in axial, coronal, and sagittal views.

**Features:**

- Multi-planar (axial/coronal/sagittal) slice views
- Color-coded tumor regions (Tumor Core: Red, Edema: Green)
- Semi-transparent overlays on brain images
- Side-by-side model comparisons
- Metrics table generation

**Usage:**

```bash
python visualize.py
```

**Output:**

- Individual model comparisons: `{model}_comparison.png`
- All models comparison: `all_models_comparison.png`
- Metrics comparison table in console

### 2. `visualize_3d.py` - 3D Renderings with PyVista

Creates 3D volumetric visualizations of tumor segmentations.

**Features:**

- 3D mesh rendering of tumor regions
- Interactive 3D views (if not saving)
- Rotating animations (GIF)
- Side-by-side 3D comparisons
- Brain volume context with transparency

**Usage:**

```bash
python visualize_3d.py
```

**Output:**

- Individual 3D renders: `{model}_3d.png`
- 3D comparison: `all_models_3d_comparison.png`
- Rotating animation: `best_model_3d_animation.gif`

## Customization

### Loading Your Own Models

Edit the model paths in the scripts:

```python
models = {
    'Baseline': visualizer.load_model("path/to/baseline.pth", 'none'),
    'SE-UNet': visualizer.load_model("path/to/se.pth", 'se'),
    'CBAM-UNet': visualizer.load_model("path/to/cbam.pth", 'cbam'),
    'Hybrid': visualizer.load_model("path/to/hybrid.pth", 'hybrid')
}
```

### Loading Your Data

Replace the placeholder data loading:

```python
# Replace this with your actual data loading
sample_image = np.random.rand(128, 128, 128, 4)  # Your MRI data
sample_gt = np.random.randint(0, 3, (128, 128, 128))  # Your ground truth
```

### Custom Slice Selection

Change the slice index for visualization:

```python
slice_idx = 64  # Middle slice (adjust as needed)
```

### Color Customization

Modify colors in the visualizer classes:

```python
self.colors = {
    'tumor_core': 'red',
    'edema': 'green',
    'enhancing': 'magenta',
    'whole_tumor': 'lightblue'
}
```

## Output Descriptions

### 2D Visualizations (`visualize.py`)

- **Individual Comparisons**: Shows ground truth vs prediction for each model
- **Multi-Model Comparison**: All models side-by-side with ground truth
- **Metrics Table**: Dice and HD95 scores for WT, TC, ET regions

### 3D Visualizations (`visualize_3d.py`)

- **3D Renders**: Volumetric tumor segmentation with brain context
- **Comparison Views**: Multiple models in subplots with linked cameras
- **Animations**: Rotating 3D views for presentations

## BraTS Region Mapping

The visualization handles the BraTS label mapping:

- **Background**: 0
- **Tumor Core**: 1 (includes original labels 1+4)
- **Edema**: 2 (original label 2)
- **Enhancing Tumor**: Part of Tumor Core in 3-class setup

## Paper-Ready Features

- **High DPI**: 300 DPI for publication quality
- **Consistent Styling**: Professional color schemes and layouts
- **Legend Integration**: Clear labeling of tumor regions
- **Metrics Integration**: Quantitative comparisons alongside visuals

## Troubleshooting

### PyVista Issues

If PyVista doesn't display properly:

```bash
export PYVISTA_OFF_SCREEN=False  # For interactive display
export PYVISTA_JUPYTER_BACKEND=trame  # For Jupyter notebooks
```

### Memory Issues

For large volumes, consider:

- Reducing image resolution
- Processing one slice at a time
- Using CPU instead of GPU for visualization

### Missing Dependencies

```bash
# For PyVista
pip install pyvista[all]

# For medical image formats
pip install nibabel
```

## Integration with Training Pipeline

After training your models, you can automatically generate visualizations:

```bash
# Train models
python train.py --attention none --outdir outputs/baseline
python train.py --attention hybrid --outdir outputs/hybrid

# Generate visualizations
python visualize.py
python visualize_3d.py
```

This creates a complete set of figures for your paper comparing baseline vs attention mechanisms.
