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
    if number <= 0:
        return "It is impossible"
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