import numpy as np
import matplotlib.pyplot as plt
import itertools

def torgerson_mds(distance_matrix):
    n = distance_matrix.shape[0]

    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * np.dot(np.dot(H, distance_matrix**2), H)
    
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    dim = np.sum(eigenvalues > 0)
    
    positive_eigenvalues = eigenvalues[:dim]
    selected_eigenvectors = eigenvectors[:, :dim]
    
    scaling = np.sqrt(np.maximum(positive_eigenvalues, 0))
    coordinates = selected_eigenvectors * scaling
    return coordinates

def find_max_embedding(A, eps=1e-3):
    n = len(A)
    max_embedded_points = []

    for r in range(4, n + 1):
        for indices in itertools.combinations(range(n), r):
            can_embed = True
            for subset in itertools.combinations(indices, 4):
                i1, i2, i3, i4 = subset
                d12 = A[i1][i2]
                d13 = A[i1][i3]
                d14 = A[i1][i4]
                d23 = A[i2][i3]
                d24 = A[i2][i4]
                d34 = A[i3][i4]

                V = abs(
                    d12**2 * d34**2 * (d14**2 + d13**2 + d24**2 + d23**2 - d12**2 - d34**2) +
                    d14**2 * d23**2 * (d12**2 + d13**2 + d24**2 + d34**2 - d14**2 - d23**2) +
                    d13**2 * d24**2 * (d12**2 + d14**2 + d34**2 + d23**2 - d13**2 - d24**2) -
                    d12**2 * d14**2 * d24**2 - d14**2 * d13**2 * d34**2 - d12**2 * d13**2 * d23**2 - d24**2 * d34**2 * d23**2
                )
                V = V / 144
                V = V ** 0.5

                if V >= eps:
                    can_embed = False
                    break
            if can_embed:
                max_embedded_points = indices

    if can_embed:
        print('Неискаженное изображение можно построить')
        return True, max_embedded_points
    else:
        print("Неискаженное изображение построить невозможно")
        return False, max_embedded_points


def load_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        matrix = []
        for line in file:
            row = list(map(float, line.split()))  # Используем float вместо int
            matrix.append(row)
    return np.array(matrix)

def build_2d_representation(num_objects, distance_matrix):
    if num_objects < 2:
        print("Число объектов должно быть не менее 2")
        return

    points = np.array([[0, 0], [distance_matrix[0, 1], 0]])

    while len(points) < num_objects:
        x3 = (-distance_matrix[1, 2]**2 + distance_matrix[0, 2]**2 + distance_matrix[0, 1]**2) / (2 * distance_matrix[0, 1])
        y3 = np.sqrt(distance_matrix[0, 2]**2 - ((-distance_matrix[1, 2]**2 + distance_matrix[0, 2]**2 + distance_matrix[0, 1]**2) / (2 * distance_matrix[0, 1]))**2)
        points = np.append(points, [[x3, y3]], axis=0)

        for i in range(3, num_objects):
            c = (distance_matrix[2, i]**2 - x3**2 - y3**2 - distance_matrix[0, i]**2) / -2
            y4 = (2 * y3 * c + np.sqrt(4 * c**2 * y3**2 - 4 * (y3**2 + x3**2) * (c**2 - distance_matrix[0, i]**2 * x3**2))) / (2 * (y3**2 + x3**2))
            y4_prime = (2 * y3 * c - np.sqrt(4 * c**2 * y3**2 - 4 * (y3**2 + x3**2) * (c**2 - distance_matrix[0, i]**2 * x3**2))) / (2 * (y3**2 + x3**2))
            x4 = (c - y4 * y3) / x3
            x4_prime = (c - y4_prime * y3) / x3

            dist_to_A4 = np.sqrt((x4 - points[1, 0])**2 + (y4 - points[1, 1])**2)
            dist_to_A4_prime = np.sqrt((x4_prime - points[1, 0])**2 + (y4_prime - points[1, 1])**2)
            if np.isclose(dist_to_A4, distance_matrix[1, i]):
                points = np.append(points, [[x4, y4]], axis=0)
            elif np.isclose(dist_to_A4_prime, distance_matrix[1, i]):
                points = np.append(points, [[x4_prime, y4_prime]], axis=0)

    return points

def plot_2d_representation(points):
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], marker='o', zorder=2)  # Установка zorder на 2
    for i, (x, y) in enumerate(points):
        plt.text(x, y + 0.1, f'A{subscript(i+1)}', fontsize=12, ha='center')
        print(f'A{subscript(i+1)}:', f'{round(x, 3), round(y, 3)}')

    for x in plt.gca().get_xticks():
        plt.axvline(x, color='gray', linestyle='-', linewidth=0.5, zorder=1)
    for y in plt.gca().get_yticks():
        plt.axhline(y, color='gray', linestyle='-', linewidth=0.5, zorder=1)

    plt.title('Изображение на плоскости')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

def vector_between_points(point1, point2):
    return np.array(point2) - np.array(point1)

def calculate_x_y(A1, A4, e1, e2):
    A1A4 = vector_between_points(A1, A4)
    x = np.dot(A1A4, e1)
    y = np.dot(A1A4, e2)
    return x, y

def subscript(n):
    subscript_digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(n).translate(subscript_digits)

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def calculate_e1_e2(A1, A2, A3):
    A1A2 = vector_between_points(A1, A2)
    A1A3 = vector_between_points(A1, A3)
    e1 = normalize_vector(A1A2)
    z = A1A3 - np.dot(A1A3, A1A2) / np.dot(A1A2, A1A2) * A1A2
    e2 = normalize_vector(z)
    return e1, e2



if __name__ == "__main__":
    distance_matrix = load_matrix_from_file('data.txt')
    num_objects = distance_matrix.shape[0]

    can_emb, mss = find_max_embedding(distance_matrix)

    if not mss:
        mss = [0, 1, 2]

    if can_emb:
        result_points = build_2d_representation(num_objects, distance_matrix)
    else:
        coords = torgerson_mds(distance_matrix)
        result_points = np.zeros((num_objects, 2))
        for i in range(num_objects):
            A1, A2, A3 = coords[mss[0]], coords[mss[1]], coords[mss[2]]
            A4 = coords[i]
            e1, e2 = calculate_e1_e2(A1, A2, A3)
            x, y = calculate_x_y(A1, A4, e1, e2)
            result_points[i] = np.array([x, y])

    plot_2d_representation(result_points)