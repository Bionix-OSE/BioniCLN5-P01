
from pathlib import Path
import re

# ============================================================
# CẤU HÌNH
# ============================================================

EXISTING_FILE = Path("ids_expanded.txt")
LV0_FILE = Path("ids_lv0.txt")
OUTPUT_FILE = Path("ids_expanded_updated.txt")
LOG_FILE = Path("ids_update_warnings.txt")


# ============================================================
# 1. NHẬN DIỆN TOÁN TỬ IDS
# ============================================================

def is_ids_operator(char):
    code = ord(char)
    return (
        0x2FF0 <= code <= 0x2FFF
        or code == 0x31EF
    )


# ============================================================
# 2. ĐỌC FILE IDS ĐÃ CÓ
# ============================================================

def load_existing_ids(file_path):
    """
    Đọc file định dạng:
    Unicode<TAB>Ký tự<TAB>IDS

    Trả về:
    - Danh sách các dòng gốc để giữ nguyên dữ liệu.
    - Dictionary ánh xạ ký tự -> IDS.
    - Tập hợp ký tự đã tồn tại.
    """
    lines = []
    decomposition_map = {}
    existing_chars = set()

    with file_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            raw = line.rstrip("\r\n")
            lines.append(raw)

            if not raw.strip():
                continue

            parts = raw.split("\t", 2)
            if len(parts) != 3:
                continue

            _, char, ids = parts
            char = char.strip()

            if char:
                existing_chars.add(char)
                decomposition_map[char] = ids

    return lines, decomposition_map, existing_chars


# ============================================================
# 3. CHUẨN HÓA CÚ PHÁP IDS_LV0
# ============================================================

def normalize_ids(raw_ids, char, warnings):
    """
    Quy tắc:
    - Chọn cách phân rã đầu tiên.
    - Dấu ; phân cách các biến thể.
    - Bỏ dấu chấm dùng làm dấu phân cách.
    - Bỏ hậu tố/chú giải trong ngoặc đơn: (K), (d), (.).
    - Bỏ nội dung chú giải trong ngoặc vuông: [b].
    - Bỏ dấu # và phần chú giải #(H), #(T), ...
    - Bỏ dấu gạch chéo ngược cuối dòng.
    - Giữ nội dung bên trong ngoặc nhọn, bỏ hai dấu ngoặc.
    - Nếu không còn IDS hợp lệ, dùng chính ký tự làm IDS.
    """

    # Chỉ lấy phương án đầu tiên trước dấu ;
    ids = raw_ids.split(";", 1)[0].strip()

    # Loại bỏ dấu \ dùng ở cuối dòng trong định dạng nguồn
    ids = ids.rstrip("\\").strip()

    # Bỏ chú giải trong ngoặc vuông, ví dụ [b]
    ids = re.sub(r"\[[^\]]*\]", "", ids)

    # Bỏ các chú giải/hậu tố trong ngoặc đơn:
    # (.), (K), (d), (H), (t), ...
    ids = re.sub(r"\([^)]*\)", "", ids)

    # Bỏ dấu # đánh dấu cú pháp mở rộng
    ids = ids.replace("#", "")

    # Bỏ dấu chấm dùng để phân cách các thành phần
    ids = ids.replace(".", "")

    # Giữ nội dung nhưng bỏ ngoặc nhọn, ví dụ {一} -> 一
    ids = ids.replace("{", "").replace("}", "")

    ids = ids.strip()

    # Chỉ chấp nhận IDS chứa ít nhất một toán tử IDS.
    # Nếu không có toán tử sau chuẩn hóa, không thể chắc
    # rằng đây là một phép phân rã hợp lệ.
    if not any(is_ids_operator(c) for c in ids):
        if ids and ids != char:
            warnings.append(
                f"{char} (U+{ord(char):04X}): "
                f"IDS sau chuẩn hóa không có toán tử: {ids!r}; "
                f"dùng chính ký tự làm IDS."
            )
        return char

    return ids


# ============================================================
# 4. ĐỌC IDS_LV0 VÀ LẤY CÁC KÝ TỰ MỚI
# ============================================================

def load_new_ids(file_path, existing_chars, warnings):
    """
    Đọc ids_lv0.txt.
    Chỉ lấy ký tự chưa xuất hiện trong ids_expanded.txt.

    Với mỗi dòng, chọn trường IDS đầu tiên sau ký tự,
    rồi chọn phương án đầu tiên trước dấu ;.
    """
    new_ids = {}
    seen_in_lv0 = set()

    with file_path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\r\n")

            if not line.strip():
                continue

            # Không dùng split() thông thường vì IDS có thể
            # chứa dấu cách; chỉ phân tách theo tab.
            parts = line.split("\t")

            if len(parts) < 2:
                warnings.append(
                    f"Dòng {line_number}: không đủ cột: {line!r}"
                )
                continue

            char = parts[0].strip()

            if not char:
                continue

            # File được giả định chứa một ký tự Unicode
            # ở cột đầu tiên.
            if len(char) != 1:
                warnings.append(
                    f"Dòng {line_number}: cột ký tự có "
                    f"{len(char)} code point: {char!r}"
                )
                continue

            # Không thêm ký tự đã có trong file hiện tại
            if char in existing_chars:
                continue

            # Tránh thêm trùng nếu ids_lv0 có nhiều dòng
            # cho cùng một ký tự.
            if char in seen_in_lv0:
                continue

            seen_in_lv0.add(char)

            # Chọn IDS ở cột đầu tiên có nội dung.
            raw_ids = next(
                (field.strip() for field in parts[1:]
                 if field.strip()),
                ""
            )

            if not raw_ids:
                warnings.append(
                    f"Dòng {line_number}: {char!r} không có IDS."
                )
                new_ids[char] = char
                continue

            normalized = normalize_ids(
                raw_ids, char, warnings
            )

            new_ids[char] = normalized

    return new_ids


# ============================================================
# 5. PHÂN RÃ ĐỆ QUY ĐẾN MỨC TỐI GIẢN
# ============================================================

def expand_character(char, decomposition_map, stack=None):
    """
    Phân rã một ký tự nếu nó có IDS trong dictionary.
    Giữ nguyên toán tử IDS và dừng khi gặp chu trình.
    """
    if is_ids_operator(char):
        return char

    if stack is None:
        stack = set()

    if char in stack:
        return char

    ids = decomposition_map.get(char)

    # Không có định nghĩa hoặc định nghĩa chính nó
    if not ids or ids == char:
        return char

    next_stack = stack | {char}

    return "".join(
        expand_character(component, decomposition_map, next_stack)
        for component in ids
    )


# ============================================================
# 6. CHƯƠNG TRÌNH CHÍNH
# ============================================================

def main():
    for path in (EXISTING_FILE, LV0_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file: {path.resolve()}"
            )

    # Đọc dữ liệu cũ
    original_lines, existing_map, existing_chars = (
        load_existing_ids(EXISTING_FILE)
    )

    warnings = []

    # Đọc và chuẩn hóa các ký tự mới
    new_map = load_new_ids(
        LV0_FILE,
        existing_chars,
        warnings
    )

    # Gộp map cũ và mới để xử lý được cả trường hợp
    # một ký tự mới phụ thuộc vào ký tự mới khác.
    combined_map = {**existing_map, **new_map}

    # Phân rã tối giản các IDS mới được thêm
    expanded_new_map = {}

    for char, ids in new_map.items():
        expanded_new_map[char] = "".join(
            expand_character(component, combined_map)
            for component in ids
        )

    # Giữ nguyên toàn bộ các dòng gốc của file cũ,
    # chỉ nối thêm những ký tự mới.
    with OUTPUT_FILE.open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        for line in original_lines:
            f.write(line + "\n")

        for char, ids in expanded_new_map.items():
            codepoint = f"U+{ord(char):04X}"
            f.write(f"{codepoint}\t{char}\t{ids}\n")

    # Ghi cảnh báo để dễ rà soát dữ liệu đặc biệt
    with LOG_FILE.open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        if warnings:
            f.write("\n".join(warnings) + "\n")
        else:
            f.write("Không có cảnh báo.\n")

    print("Hoàn tất!")
    print(f"Ký tự đã có: {len(existing_chars)}")
    print(f"Ký tự mới được bổ sung: {len(new_map)}")
    print(f"File kết quả: {OUTPUT_FILE.resolve()}")
    print(f"File cảnh báo: {LOG_FILE.resolve()}")


if __name__ == "__main__":
    main()



