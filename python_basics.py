"""Заготовки задач на базовый Python."""
from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput

def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = "aeiouAEIOU"
    return sum(1 for char in text if char in vowels)

def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    seen = set()
    for char in text:
        if char in seen:
            return False
        seen.add(char)
    return True

def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count('1')

def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    if number < 10:
        return 0
    count = 0
    while number >= 10:
        digits = [int(d) for d in str(number)]
        product = 1
        for d in digits:
            product *= d
        number = product
        count += 1
    return count

def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    if n != len(expected):
        raise ValueError("Векторы разной длины")
    return sum((p - e) ** 2 for p, e in zip(predicted, expected)) / n

def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    if number <= 1:
        return ""
    factors = {}
    d = 2
    while d * d <= number:
        while number % d == 0:
            factors[d] = factors.get(d, 0) + 1
            number //= d
        d += 1
    if number > 1:
        factors[number] = factors.get(number, 0) + 1
    
    result = []
    for prime in sorted(factors.keys()):
        power = factors[prime]
        if power == 1:
            result.append(f"({prime})")
        else:
            result.append(f"({prime}**{power})")
    return "".join(result)

def pyramid(data: PositiveIntegerInput) -> int | str:
    number = data.value
    n = 0
    total = 0
    while total < number:
        n += 1
        total += n * n
    if total == number:
        return n
    return "It is impossible"

def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    s = str(number)
    n = len(s)
    if n % 2 == 1:
        mid = n // 2
        left = s[:mid]
        right = s[mid + 1:]
    else:
        mid_start = n // 2 - 1
        left = s[:mid_start]
        right = s[mid_start + 2:]
        
    if not left or not right:
        return True
        
    left_sum = sum(int(d) for d in left)
    right_sum = sum(int(d) for d in right)
    return left_sum == right_sum


if __name__ == "__main__":
    
    # 1. Подсчёт гласных
    assert count_vowels(TextInput("hello world")) == 3
    assert count_vowels(TextInput("AEIOU")) == 5
    assert count_vowels(TextInput("bcdfg")) == 0

    # 2. Уникальные символы
    assert has_unique_characters(TextInput("abcdef")) is True
    assert has_unique_characters(TextInput("aabc")) is False
    assert has_unique_characters(TextInput("aA")) is True  # регистр учитывается

    # 3. Единичные биты
    assert count_one_bits(PositiveIntegerInput(7)) == 3
    assert count_one_bits(PositiveIntegerInput(8)) == 1
    assert count_one_bits(PositiveIntegerInput(15)) == 4

    # 4. Мультипликативная устойчивость
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4

    # 5. Средняя квадратичная ошибка
    assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0.0
    assert mse(VectorPairInput([1, 2, 3], [2, 3, 4])) == 1.0
    assert mse(VectorPairInput([0, 0], [1, 1])) == 1.0

    # 6. Разложение на простые множители
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(12)) == "(2**2)(3)"
    assert prime_factorization(PositiveIntegerInput(7)) == "(7)"

    # 7. Пирамида из кубиков
    assert pyramid(PositiveIntegerInput(1)) == 1
    assert pyramid(PositiveIntegerInput(5)) == 2
    assert pyramid(PositiveIntegerInput(14)) == 3
    assert pyramid(PositiveIntegerInput(6)) == "It is impossible"

    # 8. Сбалансированное число
    assert is_balanced_number(PositiveIntegerInput(1234006)) is True
    assert is_balanced_number(PositiveIntegerInput(123456)) is False
    assert is_balanced_number(PositiveIntegerInput(1221)) is True