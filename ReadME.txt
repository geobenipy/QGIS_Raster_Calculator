# QGIS Raster Processor Plugin 
# By benedikt.haimerl@uni-Hamburg.de
## Installation and User Guide

### What does this plugin do?
This plugin helps you process very large raster files and point clouds efficiently:
- **Interpolate point clouds** to create raster surfaces (DEMs, etc.)
- **Resample rasters** to different resolutions
- **Smooth rasters** to reduce noise

The plugin is optimized for files up to 100+ GB and uses minimal RAM.

---

## Installation Instructions

### Step 1: Find your QGIS plugins folder

**Windows:**
1. Press `Windows Key + R`
2. Type: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins`
3. Press Enter
4. If the folder doesn't exist, create it

**Mac:**
1. Get Windows

### Step 2: Create the plugin folder
- Copy raster_processor foder onto Plugins Folder

### Step 3: Install required Python packages

1. **Open OSGeo4W Shell** 

2. **Windows users:**
   - Find "OSGeo4W Shell" in your Start Menu (installed with QGIS)
   - Right-click and "Run as administrator"

3. **Run these commands** (one at a time):

```bash
python -m pip install --upgrade pip
python -m pip install scipy
python -m pip install numpy
```

4. Wait for installation to complete 

### Step 4: Activate the plugin in QGIS

1. **Open QGIS**
2. Go to menu: **Plugins → Manage and Install Plugins**
3. Click on **Installed** tab
4. Find **"Advanced Raster Processor"** in the list
5. **Check the box** next to it to activate
6. Click **Close**

You should now see a new menu item: **Plugins → Raster Processor → Advanced Raster Processor**

---

## How to Use the Plugin

### Opening the Plugin

1. Go to menu: **Plugins → Raster Processor → Advanced Raster Processor**
2. A dialog window will open

---

## Function 1: Point Cloud Interpolation

**Use this to:** Convert point data (XYZ points) into a continuous raster surface (like creating a DEM from LiDAR points)

### Steps:

1. **Select "Point Cloud Interpolation"** (top radio button)

2. **Choose your point layer:**
   - Click the dropdown under "Punktlayer" (Point Layer)
   - Select your vector layer with points
   - The layer must already be loaded in QGIS

3. **Choose the Z-value field:**
   - Click the dropdown under "Z-Wert Feld" (Z-Value Field)
   - Select the field containing elevation/height values

4. **Choose interpolation method:**
   - **IDW (fast)** - Best for large datasets, very fast
   - **Cubic** - Smooth results, medium speed
   - **Linear** - Simple interpolation, fast
   - **Nearest** - Uses closest point value, fastest
   - **RBF (precise)** - Most accurate, slower

5. **Set resolution:**
   - Enter pixel size in meters (e.g., 1.0 = 1 meter pixels)
   - Smaller = more detailed but larger file size

6. **Set output file:**
   - Click the **"..."** button
   - Choose where to save the result
   - Give it a name like `interpolated_surface.tif`

7. **Processing options:**
   - Check "Parallel Processing" for faster processing (recommended)
   - Adjust number of cores (use CPU count minus 1)

8. **Click "Verarbeiten"** (Process)

---

## Function 2: Raster Resampling

**Use this to:** Change the resolution of an existing raster (make it finer or coarser)

### Steps:

1. **Select "Raster Resample"** (middle radio button)

2. **Choose your raster:**
   - Click the dropdown under "Raster"
   - Select the raster layer you want to resample

3. **Set resample factor:**
   - **2.0** = double the resolution (4x more pixels)
   - **0.5** = half the resolution (1/4 pixels)
   - Example: 10m → 5m pixels use factor 2.0

4. **Choose method:**
   - **Bilinear** - Good for continuous data (DEMs)
   - **Cubic** - Smoother results
   - **Nearest** - For categorical data (land use maps)
   - **Average** - Good for downsampling
   - **Lanczos** - Highest quality, slower

5. **Set output file** (click "..." button)

6. **Click "Verarbeiten"** (Process)

---

## Function 3: Raster Smoothing

**Use this to:** Reduce noise or "roughness" in a raster

### Steps:

1. **Select "Raster Smoothing"** (bottom radio button)

2. **Choose your raster:**
   - Click the dropdown under "Raster"
   - Select the raster layer you want to smooth

3. **Set smooth factor:**
   - **1.0** = light smoothing
   - **2.0** = medium smoothing
   - **5.0+** = heavy smoothing
   - Higher values = more smoothing effect

4. **Choose method:**
   - **Gaussian** - Natural-looking smooth (recommended)
   - **Uniform (Mean)** - Simple averaging filter

5. **Set output file** (click "..." button)

6. **Click "Verarbeiten"** (Process)

---

## Tips for Large Files (50+ GB)

### For Point Cloud Interpolation:
- Use **IDW method** - it's the fastest
- Start with **coarser resolution** (5m or 10m) to test
- Enable **Parallel Processing**
- Be patient - 100 GB files may take 30+ minutes

### For Resampling:
- Works very efficiently even on huge files
- The plugin uses streaming - won't crash your computer

### For Smoothing:
- Larger smooth factors take longer
- Very large files (100+ GB) may take 10-20 minutes
- Progress is shown in the progress bar and messages

---

## Understanding the Progress

While processing, you'll see:
- **Progress bar** showing percentage complete
- **Status messages** at the top of QGIS (blue bar)
- Messages like "Processing tile 50/200" - this is normal!

---

## Troubleshooting

### Problem: Plugin doesn't appear in menu
**Solution:** 
- Make sure files are in the correct folder
- Restart QGIS completely
- Check Plugin Manager that it's activated

### Problem: "Module not found: scipy"
**Solution:**
- Reopen OSGeo4W Shell as administrator
- Run: `python -m pip install scipy numpy`

### Problem: "Out of memory" error
**Solution:**
- Close other programs
- Use coarser resolution for interpolation
- Try smaller smooth factors
- The plugin should handle 100GB files with 16GB RAM

### Problem: Processing is very slow
**Solution:**
- Enable "Parallel Processing"
- For interpolation: use IDW method instead of RBF
- Use coarser resolution
- Large files naturally take longer - be patient!

### Problem: Output file is too large
**Solution:**
- Use coarser resolution for interpolation
- The plugin automatically uses LZW compression
- Consider if you need such high resolution

---

## Example Workflow

### Creating a DEM from LiDAR points:

1. Load your point cloud shapefile/geopackage in QGIS
2. Open the plugin
3. Select "Point Cloud Interpolation"
4. Choose your layer and elevation field
5. Method: IDW (fast)
6. Resolution: 1.0 (for 1 meter pixels)
7. Click Process
8. Wait for completion
9. Result appears as new layer in QGIS

### Smoothing a noisy DEM:

1. Load your DEM raster in QGIS
2. Open the plugin
3. Select "Raster Smoothing"
4. Choose your DEM layer
5. Method: Gaussian
6. Factor: 2.0
7. Click Process
8. Compare original vs smoothed result

---

## Support

If you have issues:
1. Check the troubleshooting section above
2. Check QGIS message log: View → Panels → Log Messages
3. Verify your input files are valid
4. Try with a smaller test file first

---

## Technical Details

**Memory Usage:**
- 10 GB file: ~2-4 GB RAM
- 50 GB file: ~3-6 GB RAM  
- 100 GB file: ~4-8 GB RAM

**Supported Formats:**
- Input: Any GDAL-supported format (GeoTIFF, etc.)
- Output: GeoTIFF (.tif)

**Performance:**
- Uses tile-based processing
- Streaming for minimal memory usage
- Multi-core support for interpolation
- Optimized for large datasets

---

## License

This plugin is open source. Feel free to modify and share.

---