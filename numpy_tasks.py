"""Заготовки задач на NumPy."""
import numpy as np
import matplotlib.pyplot as plt
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

def plot_histograms(stats: MatrixStatistics):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].hist(stats.row_means, bins=10, color='blue', alpha=0.7)
    axes[0, 0].set_title("Row Means")
    axes[0, 1].hist(stats.column_means, bins=10, color='green', alpha=0.7)
    axes[0, 1].set_title("Column Means")
    axes[1, 0].hist(stats.row_variances, bins=10, color='red', alpha=0.7)
    axes[1, 0].set_title("Row Variances")
    axes[1, 1].hist(stats.column_variances, bins=10, color='purple', alpha=0.7)
    axes[1, 1].set_title("Column Variances")
    plt.tight_layout()
    plt.show()

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
    x0 = image_width / 2.0
    y0 = image_height / 2.0
    
    for i in range(image_height):
        for j in range(image_width):
            x_norm = (j - x0) / semi_axis_x if semi_axis_x != 0 else 0
            y_norm = (i - y0) / semi_axis_y if semi_axis_y != 0 else 0
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


if __name__ == "__main__":
    
    # 1. Сумма произведений матриц на векторы
    matrices = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    vectors = [np.array([[1], [2]]), np.array([[3], [4]])]
    data = MatrixVectorBatchInput(matrices=matrices, vectors=vectors)
    result = sum_prod(data)
    assert result.shape == (2, 1)
    assert np.array_equal(result, np.array([[44], [64]]))

    # 2. Бинаризация матрицы
    matrix = np.array([[0.3, 0.7], [1.2, -0.5]])
    data = BinarizeInput(matrix=matrix, threshold=0.5)
    result = binarize(data)
    assert np.array_equal(result, np.array([[0, 1], [1, 0]]))

    # 3. Уникальные элементы строк и столбцов
    matrix = np.array([[1, 2, 2], [4, 5, 6]])
    data = MatrixInput(matrix=matrix)
    
    rows_result = unique_rows(data)
    assert rows_result == [[1.0, 2.0], [4.0, 5.0, 6.0]]
    
    cols_result = unique_columns(data)
    assert len(cols_result) == 3

    # 4. Статистики случайной матрицы
    data = RandomMatrixInput(rows=3, columns=4, seed=42)
    stats = matrix_statistics(data)
    assert stats.matrix.shape == (3, 4)
    assert len(stats.row_means) == 3
    plot_histograms(stats)

    # 5. Шахматная матрица
    data = ChessInput(rows=3, columns=3, first=0.0, second=1.0)
    board = chess(data)
    assert board.shape == (3, 3)
    assert board[0, 0] == 0.0
    assert board[0, 1] == 1.0

    # 6. Прямоугольник и эллипс
    rect = RectangleInput(
        width=3, height=2,
        image_height=5, image_width=5,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0)
    )
    img_rect = draw_rectangle(rect)
    assert img_rect.shape == (5, 5, 3)
    
    ellipse = EllipseInput(
        semi_axis_x=2, semi_axis_y=1,
        image_height=5, image_width=5,
        shape_color=(0, 255, 0),
        background_color=(0, 0, 0)
    )
    img_ellipse = draw_ellipse(ellipse)
    assert img_ellipse.shape == (5, 5, 3)

    # 7. Анализ временного ряда
    series = TimeSeriesInput(values=[1, 3, 2, 4, 1], window=2)
    stats = analyze_time_series(series)
    assert stats.local_maxima_indices == [1, 3]
    assert stats.local_minima_indices == [2]

    # 8. One-hot encoding
    labels = np.array([0, 1, 2])
    data = OneHotInput(labels=labels, class_count=None)
    result = one_hot(data)
    assert result.shape == (3, 3)
    assert np.array_equal(result, np.eye(3, dtype=np.int64))