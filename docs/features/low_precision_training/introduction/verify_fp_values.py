#!/usr/bin/env python3
"""
Skrypt weryfikujący wartości zmiennoprzecinkowe z diagramu fp_formats_comparison.svg

Diagram przedstawia:
- FP32: 1 bit znaku + 8 bitów wykładnika + 23 bity mantysy
- BF16: 1 bit znaku + 8 bitów wykładnika + 7 bitów mantysy  
- FP16: 1 bit znaku + 5 bitów wykładnika + 10 bitów mantysy
"""

import struct


def binary_to_float(sign_bit: int, exponent_bits: str, mantissa_bits: str, 
                    exponent_bias: int, format_name: str) -> float:
    """
    Konwertuje reprezentację binarną na wartość zmiennoprzecinkową.
    
    Args:
        sign_bit: Bit znaku (0 lub 1)
        exponent_bits: String bitów wykładnika (np. "01111101")
        mantissa_bits: String bitów mantysy (np. "10010100...")
        exponent_bias: Wartość biasu dla wykładnika
        format_name: Nazwa formatu do wyświetlenia
    
    Returns:
        Obliczona wartość float
    """
    # Oblicz wykładnik
    exponent_value = int(exponent_bits, 2)
    actual_exponent = exponent_value - exponent_bias
    
    # Oblicz mantysę (implicit leading 1 dla znormalizowanych liczb)
    mantissa_value = 1.0  # Implicit leading 1
    for i, bit in enumerate(mantissa_bits):
        if bit == '1':
            mantissa_value += 2 ** (-(i + 1))
    
    # Oblicz końcową wartość
    sign = (-1) ** sign_bit
    value = sign * mantissa_value * (2 ** actual_exponent)
    
    print(f"\n{format_name}:")
    print(f"  Znak: {sign_bit} ({'ujemny' if sign_bit else 'dodatni'})")
    print(f"  Wykładnik binarny: {exponent_bits} = {exponent_value}")
    print(f"  Wykładnik rzeczywisty: {exponent_value} - {exponent_bias} = {actual_exponent}")
    print(f"  Mantysa binarna: {mantissa_bits}")
    print(f"  Mantysa wartość: 1 + {mantissa_value - 1:.10f} = {mantissa_value:.10f}")
    print(f"  Obliczona wartość: {sign} × {mantissa_value:.10f} × 2^{actual_exponent} = {value:.10f}")
    
    return value


def verify_with_struct():
    """Weryfikacja przy użyciu struct dla FP32."""
    print("\n" + "="*70)
    print("WERYFIKACJA Z UŻYCIEM struct (FP32)")
    print("="*70)
    
    # FP32 bity z SVG: sign=0, exp=01111101, mantissa=10010100101011110101000
    fp32_bits = "0" + "01111101" + "10010100101011110101000"
    fp32_int = int(fp32_bits, 2)
    fp32_bytes = struct.pack('>I', fp32_int)
    fp32_value = struct.unpack('>f', fp32_bytes)[0]
    
    print(f"  FP32 bity: {fp32_bits}")
    print(f"  FP32 jako int: {fp32_int}")
    print(f"  FP32 wartość (struct): {fp32_value:.10f}")
    
    return fp32_value


def main():
    print("="*70)
    print("WERYFIKACJA WARTOŚCI Z DIAGRAMU fp_formats_comparison.svg")
    print("="*70)
    
    # Dane z SVG
    # FP32: sign=0, exponent=01111101, mantissa=10010100101011110101000
    fp32_sign = 0
    fp32_exponent = "01111101"
    fp32_mantissa = "10010100101011110101000"
    fp32_claimed = 0.3952
    
    # BF16: sign=0, exponent=01111101, mantissa=1001010
    bf16_sign = 0
    bf16_exponent = "01111101"
    bf16_mantissa = "1001010"
    bf16_claimed = 0.3945
    
    # FP16: sign=0, exponent=01101, mantissa=1001010010
    fp16_sign = 0
    fp16_exponent = "01101"
    fp16_mantissa = "1001010010"
    fp16_claimed = 0.3950
    
    # Weryfikacja FP32 (bias = 127)
    fp32_calculated = binary_to_float(fp32_sign, fp32_exponent, fp32_mantissa, 127, "FP32")
    
    # Weryfikacja BF16 (bias = 127, taki sam jak FP32)
    bf16_calculated = binary_to_float(bf16_sign, bf16_exponent, bf16_mantissa, 127, "BF16")
    
    # Weryfikacja FP16 (bias = 15)
    fp16_calculated = binary_to_float(fp16_sign, fp16_exponent, fp16_mantissa, 15, "FP16")
    
    # Weryfikacja z struct
    fp32_struct = verify_with_struct()
    
    # Podsumowanie
    print("\n" + "="*70)
    print("PODSUMOWANIE")
    print("="*70)
    
    print(f"\n{'Format':<10} {'Podana':<12} {'Obliczona':<15} {'Różnica':<12} {'Status'}")
    print("-" * 60)
    
    results = [
        ("FP32", fp32_claimed, fp32_calculated),
        ("BF16", bf16_claimed, bf16_calculated),
        ("FP16", fp16_claimed, fp16_calculated),
    ]
    
    all_ok = True
    for name, claimed, calculated in results:
        diff = abs(claimed - calculated)
        # Tolerancja dla zaokrąglenia wyświetlania (4 miejsca po przecinku)
        ok = diff < 0.0001
        status = "✓ OK" if ok else "✗ BŁĄD"
        if not ok:
            all_ok = False
        print(f"{name:<10} {claimed:<12.4f} {calculated:<15.10f} {diff:<12.6f} {status}")
    
    print("\n" + "-" * 60)
    print(f"FP32 weryfikacja struct: {fp32_struct:.10f}")
    
    if all_ok:
        print("\n✓ Wszystkie wartości w SVG są POPRAWNE!")
    else:
        print("\n✗ Niektóre wartości w SVG są NIEPOPRAWNE!")
    
    # Dodatkowa analiza - sprawdźmy dokładne wartości
    print("\n" + "="*70)
    print("DOKŁADNE WARTOŚCI (więcej miejsc po przecinku)")
    print("="*70)
    print(f"FP32 dokładna wartość: {fp32_calculated:.15f}")
    print(f"BF16 dokładna wartość: {bf16_calculated:.15f}")
    print(f"FP16 dokładna wartość: {fp16_calculated:.15f}")


if __name__ == "__main__":
    main()

