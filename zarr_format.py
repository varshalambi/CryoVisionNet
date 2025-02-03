import zarr
import matplotlib.pyplot as plt
import numpy as np

# Load the Zarr store
store = zarr.open('/Users/varsha/Downloads/czii-cryo-et-object-identification/test/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr', mode='r')
print(type(store['0']))
print(store['0'].shape)

''''

# Access each array by key
array_0 = store['0']
array_1 = store['1']
array_2 = store['2']

# Access a slice from each array
slice_0 = array_0[0, :, :]
slice_1 = array_1[0, :, :]
slice_2 = array_2[0, :, :]

# Setup a figure with multiple subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Display each slice in a subplot
axes[0].imshow(slice_0, cmap='gray')
axes[0].set_title('Slice from Array 0')
axes[1].imshow(slice_1, cmap='gray')
axes[1].set_title('Slice from Array 1')
axes[2].imshow(slice_2, cmap='gray')
axes[2].set_title('Slice from Array 2')

plt.tight_layout()
plt.show()

# Function to visualize a 3D slice along different axes
def plot_slices(x, y, z):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(zarr_data[x, :, :], cmap='gray')
    axes[0].set_title('Slice along first dimension')
    axes[1].imshow(zarr_data[:, y, :], cmap='gray')
    axes[1].set_title('Slice along second dimension')
    axes[2].imshow(zarr_data[:, :, z], cmap='gray')
    axes[2].set_title('Slice along third dimension')
    plt.show()

# Visualize slices at the center
center_x, center_y, center_z = zarr_data.shape[0] // 2, zarr_data.shape[1] // 2, zarr_data.shape[2] // 2
plot_slices(center_x, center_y, center_z)



from skimage import filters, measure, morphology

# Example: Apply a Gaussian filter to smooth the image
smoothed = filters.gaussian(slice_0, sigma=1)

# Threshold the image to create a binary image
binary_image = smoothed > filters.threshold_otsu(smoothed)

# Label connected regions
labeled_image, num_features = measure.label(binary_image, return_num=True)
print(f"Detected {num_features} features")

# Display the processed image
plt.imshow(labeled_image, cmap='nipy_spectral')
plt.colorbar()
plt.show()


properties = measure.regionprops_table(labeled_image, intensity_image=slice_0, properties=('area', 'mean_intensity', 'solidity'))

# Convert to a DataFrame for easier manipulation
import pandas as pd
df = pd.DataFrame(properties)
print(df.head())


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Example: Assuming you have features and labels
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
print("Accuracy:", model.score(X_test, y_test))'''