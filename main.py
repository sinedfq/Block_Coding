import numpy as np
from itertools import product


def read_matrix(filename: str) -> np.ndarray:
    """Чтение порождающей матрицы из файла"""
    with open(filename, 'r') as f:
        n, m = map(int, f.readline().split())
        matrix = []
        for _ in range(n):
            row = list(map(int, f.readline().split()))
            matrix.append(row)
    return np.array(matrix)


def generate_matrix(n: int, m: int) -> np.ndarray:
    """Генерация порождающей матрицы вида G = [I_n | D]"""
    if n >= m:
        raise ValueError("n должно быть меньше m")

    # Единичная матрица I_n
    identity = np.eye(n, dtype=int)

    # Случайная матрица D размером n x (m-n)
    D = np.random.randint(0, 2, size=(n, m - n))

    # Объединяем матрицы
    G = np.hstack((identity, D))
    return G


def save_matrix(filename: str, matrix: np.ndarray):
    """Сохранение матрицы в файл"""
    n, m = matrix.shape
    with open(filename, 'w') as f:
        f.write(f"{n} {m}\n")
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")


def get_code_dimension(G: np.ndarray) -> int:
    """Определение размерности кода (количество информационных символов)"""
    return G.shape[0]


def get_code_size(G: np.ndarray) -> int:
    """Определение количества кодовых слов"""
    return 2 ** get_code_dimension(G)


def get_all_codewords(G: np.ndarray) -> list:
    """Генерация всех кодовых слов"""
    n, m = G.shape
    codewords = []

    # Генерируем все возможные информационные слова длины n
    for info_word in product([0, 1], repeat=n):
        codeword = np.mod(np.dot(info_word, G), 2)
        codewords.append(tuple(codeword))

    return codewords


def get_min_distance(codewords: list) -> int:
    """Вычисление минимального кодового расстояния"""
    min_distance = float('inf')
    n = len(codewords)

    # Сравниваем каждую пару кодовых слов
    for i in range(n):
        for j in range(i + 1, n):
            distance = sum(c1 != c2 for c1, c2 in zip(codewords[i], codewords[j]))
            if distance < min_distance:
                min_distance = distance
                if min_distance == 1:
                    return 1  # Минимальное расстояние не может быть меньше 1

    return min_distance


def analyze_code(filename: str):
    """Анализ характеристик линейного кода"""
    try:
        G = read_matrix(filename)
        print(f"\nАнализ кода из файла: {filename}")
        print("Порождающая матрица:")
        print(G)

        # Проверка формы матрицы
        n, m = G.shape
        identity_part = G[:, :n]
        if not np.array_equal(identity_part, np.eye(n, dtype=int)):
            print("Предупреждение: матрица не имеет вида [I_n | D]")

        # Характеристики кода
        dimension = get_code_dimension(G)
        size = get_code_size(G)
        codewords = get_all_codewords(G)
        min_distance = get_min_distance(codewords)

        print("\nХарактеристики кода:")
        print(f"Размерность кода (k): {dimension}")
        print(f"Длина кода (n): {m}")
        print(f"Количество кодовых слов: {size}")
        print(f"Минимальное кодовое расстояние: {min_distance}")

        return {
            'filename': filename,
            'n': m,
            'k': dimension,
            'size': size,
            'd_min': min_distance
        }
    except Exception as e:
        print(f"Ошибка при анализе файла {filename}: {str(e)}")
        return None


def generate_test_files():
    """Генерация тестовых файлов с порождающими матрицами"""
    test_sizes = [(3, 5), (4, 7), (2, 4), (3, 6), (5, 8)]

    for i, (n, m) in enumerate(test_sizes, 1):
        filename = f"matrix_{i}.txt"
        G = generate_matrix(n, m)
        save_matrix(filename, G)
        print(f"Сгенерирован файл: {filename}")


def main():
    """Основная функция программы"""
    print("Генерация тестовых файлов...")
    generate_test_files()

    print("\nАнализ кодов...")
    results = []
    for i in range(1, 6):
        filename = f"matrix_{i}.txt"
        result = analyze_code(filename)
        if result:
            results.append(result)

    # Вывод таблицы результатов
    print("\nСводная таблица результатов:")
    print("Файл         | n (длина) | k (размерность)  | Количество слов  | d_min")
    print("-------------|-----------|------------------|------------------|------")
    for res in results:
        print(f"{res['filename']:10} | {res['n']:9} | {res['k']:16} | {res['size']:16} | {res['d_min']:5}")


if __name__ == "__main__":
    main()