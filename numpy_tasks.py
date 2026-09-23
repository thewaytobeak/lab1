"""Заготовки задач на NumPy."""
import numpy as np
#import matplotlib.pyplot as plt
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)

def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    total = np.zeros_like(matrices[0] @ vectors[0])
    for mat, vec in zip(matrices, vectors):
        total += mat @ vec
    return total

def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (matrix > threshold).astype(np.int64)

def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in matrix]

def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    n_cols = matrix.shape[1]
    return [np.unique(matrix[:, col_idx]).tolist() for col_idx in range(n_cols)]

def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(mean, std, size=(rows, columns))
    
    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )

# def plot_histograms(stats: MatrixStatistics):
#     fig, axes = plt.subplots(2, 2, figsize=(10, 8))
#     axes[0, 0].hist(stats.row_means, bins=10, color='blue', alpha=0.7)
#     axes[0, 0].set_title("Row Means")
#     axes[0, 1].hist(stats.column_means, bins=10, color='green', alpha=0.7)
#     axes[0, 1].set_title("Column Means")
#     axes[1, 0].hist(stats.row_variances, bins=10, color='red', alpha=0.7)
#     axes[1, 0].set_title("Row Variances")
#     axes[1, 1].hist(stats.column_variances, bins=10, color='purple', alpha=0.7)
#     axes[1, 1].set_title("Column Variances")
#     plt.tight_layout()
#     plt.show()

def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    i, j = np.indices((rows, columns))
    return np.where((i + j) % 2 == 0, first, second).astype(np.float64)

def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    
    image = np.full((image_height, image_width, 3), background_color, dtype=np.int64)
    x_start = (image_width - width) // 2
    y_start = (image_height - height) // 2
    
    for i in range(y_start, y_start + height):
        for j in range(x_start, x_start + width):
            image[i, j] = shape_color
    return image

def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.full((image_height, image_width, 3), background_color, dtype=np.int64)
    x0 = (image_width - 1) / 2
    y0 = (image_height - 1) / 2

    for i in range(image_height):
        for j in range(image_width):
            x_norm = (j - x0) / (abs(semi_axis_x) if semi_axis_x != 0 else 1e-9)
            y_norm = (i - y0) / (abs(semi_axis_y) if semi_axis_y != 0 else 1e-9)
            if x_norm ** 2 + y_norm ** 2 <= 1:
                image[i, j] = shape_color
    return image

def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    n = len(values)
    
    local_maxima_indices = []
    local_minima_indices = []
    for i in range(1, n - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            local_maxima_indices.append(i)
        elif values[i] < values[i - 1] and values[i] < values[i + 1]:
            local_minima_indices.append(i)
            
    moving_average = np.convolve(values, np.ones(window) / window, mode='valid')
    
    return TimeSeriesStatistics(
        mean=np.mean(values),
        variance=np.var(values),
        std=np.std(values),
        local_maxima_indices=local_maxima_indices,
        local_minima_indices=local_minima_indices,
        moving_average=moving_average,
    )

def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    if class_count is None:
        class_count = int(np.max(labels)) + 1
        
    n_samples = len(labels)
    one_hot_matrix = np.zeros((n_samples, class_count), dtype=np.int64)
    for idx, label in enumerate(labels):
        one_hot_matrix[idx, label] = 1
    return one_hot_matrix