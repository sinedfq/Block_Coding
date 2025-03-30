<h1>Блочное кодирование</h1>

<h3>Цель работы:</h3>
Экспериментальное изучение свойств блочного кодирования.

<h3>Среда программирования:</h3>
1. Python

<h3>Результат: </h3>

```
1. Программа. 
2. Тестовые примеры. 
3. Отчет.
```

<h3>Задание: </h3>

```
1. Для выполнения работы необходим сгенерированный файл с
неравномерным распределением из практической работы 1
При блочном кодировании входная последовательность разбивается на блоки
равной длины, которые кодируются целиком. Поскольку вероятностное
распределение символов в файле известно, то и вероятности блоков могут
быть вычислены и использованы для построения кода.

2. Закодировать файл блочным методом кодирования (можно использовать
любой метод кодирования), размер блока n = 1,2,3,4. Вычислить избыточность
кодирования на символ входной последовательности для каждого размера
блока.

3. После тестирования программы необходимо заполнить таблицу и
проанализировать полученные результаты, сравнить с теоретическими
оценками.
```

<h3>Техническое описани кода: </h3>


```python
import numpy as np
from itertools import product
```
1. numpy — для работы с матрицами и линейной алгеброй.
2. itertools.product — для генерации всех возможных комбинаций битовых строк (информационных слов).

----

def read_matrix(filename: str) -> np.ndarray:

```python 
def read_matrix(filename: str) -> np.ndarray:
    """Чтение порождающей матрицы из файла"""
    with open(filename, 'r') as f:
        n, m = map(int, f.readline().split())
        matrix = []
        for _ in range(n):
            row = list(map(int, f.readline().split()))
            matrix.append(row)
    return np.array(matrix)
```

1. Читает матрицу из файла.
2. Первая строка файла — размеры матрицы (n строк, m столбцов).
3. Остальные строки — элементы матрицы (по строкам).
4. Возвращает матрицу в виде numpy.ndarray.

-----

def generate_matrix(n: int, m: int) -> np.ndarray:

```python
def generate_matrix(n: int, m: int) -> np.ndarray:
    """Генерация порождающей матрицы вида G = [I_n | D]"""
    if n >= m:
        raise ValueError("n должно быть меньше m")

    identity = np.eye(n, dtype=int)
    D = np.random.randint(0, 2, size=(n, m - n))
    G = np.hstack((identity, D))
    return G
```
1. Генерирует порождающую матрицу линейного кода в канонической форме [I_n | D], где:
```
I_n — единичная матрица размера n × n,

D — случайная бинарная матрица размера n × (m - n).
```
2. Если n ≥ m, вызывает ошибку.

-----

def save_matrix(filename: str, matrix: np.ndarray):

```python 
def save_matrix(filename: str, matrix: np.ndarray):
    """Сохранение матрицы в файл"""
    n, m = matrix.shape
    with open(filename, 'w') as f:
        f.write(f"{n} {m}\n")
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")
```
1.Сохраняет матрицу в файл в формате:
```
Первая строка: n m (размеры матрицы).
Последующие строки: элементы матрицы построчно.
```

----

def get_code_dimension(G: np.ndarray) -> int:  &  def get_code_size(G: np.ndarray) -> int:

```python 
def get_code_dimension(G: np.ndarray) -> int:
    """Определение размерности кода (количество информационных символов)"""
    return G.shape[0]

def get_code_size(G: np.ndarray) -> int:
    """Определение количества кодовых слов"""
    return 2 ** get_code_dimension(G)

```
1. Возвращает размерность кода (k), равную числу строк матрицы G.
2. Возвращает количество кодовых слов (2^k, где k — размерность кода).

----

def get_all_codewords(G: np.ndarray) -> list:

```python
def get_all_codewords(G: np.ndarray) -> list:
    """Генерация всех кодовых слов"""
    n, m = G.shape
    codewords = []

    for info_word in product([0, 1], repeat=n):
        codeword = np.mod(np.dot(info_word, G), 2)
        codewords.append(tuple(codeword))

    return codewords
```
1. Генерирует все кодовые слова:
```
Перебирает все возможные информационные слова длины n (все комбинации 0 и 1).
Умножает каждое информационное слово на матрицу G (по модулю 2).
```
2. Возвращает список кодовых слов.

---

def get_min_distance(codewords: list) -> int:

```python
def get_min_distance(codewords: list) -> int:
    """Вычисление минимального кодового расстояния"""
    min_distance = float('inf')
    n = len(codewords)

    for i in range(n):
        for j in range(i + 1, n):
            distance = sum(c1 != c2 for c1, c2 in zip(codewords[i], codewords[j]))
            if distance < min_distance:
                min_distance = distance
                if min_distance == 1:
                    return 1

    return min_distance
```
1. Вычисляет минимальное расстояние Хэмминга между кодовыми словами:
```
Сравнивает каждую пару слов, подсчитывая различающиеся биты.
Если расстояние равно 1, сразу возвращает его (оптимизация).
```

![image](https://github.com/user-attachments/assets/319446f5-7c99-4282-b4e2-1c1031fd9799)


---

def analyze_code(filename: str):

```python
def analyze_code(filename: str):
    """Анализ характеристик линейного кода"""
    try:
        G = read_matrix(filename)
        print(f"\nАнализ кода из файла: {filename}")
        print("Порождающая матрица:")
        print(G)

        n, m = G.shape
        identity_part = G[:, :n]
        if not np.array_equal(identity_part, np.eye(n, dtype=int)):
            print("Предупреждение: матрица не имеет вида [I_n | D]")

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
```

Анализирует код из файла:
1. Читает матрицу G.

2. Проверяет, имеет ли она каноническую форму [I_n | D].

3. Вычисляет характеристики кода:
```
Размерность (k),

Длину (n),

Количество кодовых слов,

Минимальное расстояние.

```
![image](https://github.com/user-attachments/assets/3b126aef-4418-4704-8787-d19753be7dd2)

4. Возвращает результат в виде словаря.

   ---

def generate_test_files():

```python 
def generate_test_files():
    """Генерация тестовых файлов с порождающими матрицами"""
    test_sizes = [(3, 5), (4, 7), (2, 4), (3, 6), (5, 8)]

    for i, (n, m) in enumerate(test_sizes, 1):
        filename = f"matrix_{i}.txt"
        G = generate_matrix(n, m)
        save_matrix(filename, G)
        print(f"Сгенерирован файл: {filename}")
```

1. Генерирует 5 тестовых файлов с матрицами разных размеров.

---

def main()

```python 
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

    print("\nСводная таблица результатов:")
    print("Файл         | n (длина) | k (размерность)  | Количество слов  | d_min")
    print("-------------|-----------|------------------|------------------|------")
    for res in results:
        print(f"{res['filename']:10} | {res['n']:9} | {res['k']:16} | {res['size']:16} | {res['d_min']:5}")
```
1. Генерирует тестовые файлы.
2. Анализирует каждый файл.
3. Выводит сводную таблицу с характеристиками кодов.
