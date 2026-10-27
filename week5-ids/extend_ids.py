
from pathlib import Path

INPUT_FILE = Path("ids.txt")
OUTPUT_FILE = Path("ids_expanded.txt")

# Các ký tự toán tử IDS nằm trong vùng U+2FF0–U+2FFF.
# U+31EF là toán tử IDS bổ sung.
def is_ids_operator(char):
    code = ord(char)
    return 0x2FF0 <= code <= 0x2FFF or code == 0x31EF


def load_ids(file_path):
    """Đọc file IDS và tạo ánh xạ: ký tự -> chuỗi phân rã."""
    decomposition_map = {}

    with file_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\r\n")

            if not line.strip():
                continue

            # Mỗi dòng có dạng: Unicode<TAB>Ký tự<TAB>IDS
            parts = line.split("\t", 2)

            if len(parts) != 3:
                print(
                    f"Cảnh báo: bỏ qua dòng {line_number} "
                    f"do không đúng định dạng: {line}"
                )
                continue

            unicode_code, character, decomposition = parts
            character = character.strip()

            if not character:
                continue

            # Nếu có nhiều định nghĩa cho cùng một ký tự,
            # định nghĩa xuất hiện sau cùng sẽ được sử dụng.
            decomposition_map[character] = decomposition

    return decomposition_map


def expand_character(char, decomposition_map, stack=None):
    """
    Phân rã đệ quy một ký tự đến mức không thể phân rã thêm.
    Giữ nguyên các toán tử IDS.
    """
    if stack is None:
        stack = set()

    # Toán tử IDS không phải thành phần cần phân rã.
    if is_ids_operator(char):
        return char

    # Phát hiện chu trình để tránh đệ quy vô hạn.
    if char in stack:
        return char

    decomposition = decomposition_map.get(char)

    # Không có định nghĩa hoặc định nghĩa chính nó:
    # đây là thành phần cuối cùng.
    if not decomposition or decomposition == char:
        return char

    # Thêm ký tự hiện tại vào nhánh đệ quy.
    new_stack = stack | {char}

    result = []
    for component in decomposition:
        result.append(
            expand_character(
                component,
                decomposition_map,
                new_stack
            )
        )

    return "".join(result)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file đầu vào: {INPUT_FILE.resolve()}"
        )

    decomposition_map = load_ids(INPUT_FILE)

    expanded_count = 0

    with INPUT_FILE.open("r", encoding="utf-8-sig") as source, \
         OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as output:

        for line in source:
            original_line = line.rstrip("\r\n")

            if not original_line.strip():
                output.write("\n")
                continue

            parts = original_line.split("\t", 2)

            if len(parts) != 3:
                # Giữ nguyên các dòng không đúng định dạng.
                output.write(original_line + "\n")
                continue

            unicode_code, character, original_ids = parts

            expanded_ids = "".join(
                expand_character(component, decomposition_map)
                for component in original_ids
            )

            output.write(
                f"{unicode_code}\t{character}\t{expanded_ids}\n"
            )

            if expanded_ids != original_ids:
                expanded_count += 1

    print("Hoàn tất phân rã IDS!")
    print(f"Số ký tự có IDS được mở rộng: {expanded_count}")
    print(f"File kết quả: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
