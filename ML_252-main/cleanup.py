"""
cleanup.py — Xoá toàn bộ output đã lưu, reset lại dự án về trạng thái ban đầu.
Chạy từ thư mục GỐC của project (ngang hàng với notebooks/, features/, models/):
    python cleanup.py
Hoặc xoá chọn lọc:
    python cleanup.py --keep-data      # giữ lại data/, chỉ xoá features/models/reports
    python cleanup.py --dry-run        # chỉ in ra, không xoá thật
"""

import os, shutil, argparse
from pathlib import Path

# ── Cấu hình: các thư mục / file cần xoá ──────────────────────
DIRS_TO_CLEAR = [
    "features",           # .npy files
    "models",             # .pkl, .pth files
    "reports/figures",    # .png files
]

FILES_TO_DELETE = [
    "reports/model_results.csv",
]

DATA_DIR = "data"         # chỉ xoá khi KHÔNG dùng --keep-data


def sizeof_fmt(num):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def count_dir(path):
    """Đếm số file và tổng dung lượng trong thư mục."""
    p = Path(path)
    if not p.exists():
        return 0, 0
    files = list(p.rglob("*"))
    files = [f for f in files if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)
    return len(files), total_size


def clear_directory(path, dry_run=False):
    """Xoá toàn bộ nội dung bên trong thư mục (giữ thư mục rỗng)."""
    p = Path(path)
    if not p.exists():
        print(f"   ⏭  {path}/ — không tồn tại, bỏ qua")
        return 0, 0

    n_files, size = count_dir(path)
    if n_files == 0:
        print(f"   ✅ {path}/ — đã rỗng")
        return 0, 0

    print(f"   🗑  {path}/ — {n_files} files ({sizeof_fmt(size)})")
    if not dry_run:
        for item in p.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        p.mkdir(exist_ok=True)   # giữ thư mục rỗng
    return n_files, size


def delete_file(path, dry_run=False):
    """Xoá một file cụ thể."""
    p = Path(path)
    if not p.exists():
        return 0
    size = p.stat().st_size
    print(f"   🗑  {path} ({sizeof_fmt(size)})")
    if not dry_run:
        p.unlink()
    return size


def main():
    parser = argparse.ArgumentParser(description="Reset ML project outputs")
    parser.add_argument("--keep-data", action="store_true",
                        help="Giữ lại thư mục data/ (không xoá dataset đã tải)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Chỉ in ra danh sách, không xoá thật")
    args = parser.parse_args()

    # Đảm bảo đang ở đúng project root
    if not Path("notebooks").exists():
        print("⚠️  Hãy chạy script này từ thư mục GỐC của project")
        print("   (thư mục chứa notebooks/, features/, models/)")
        return

    tag = " [DRY RUN]" if args.dry_run else ""
    print("=" * 55)
    print(f"🧹 CLEANUP PROJECT{tag}")
    print("=" * 55)

    total_files, total_size = 0, 0

    # Xoá features/, models/, reports/figures/
    for d in DIRS_TO_CLEAR:
        n, s = clear_directory(d, args.dry_run)
        total_files += n
        total_size  += s

    # Xoá file lẻ
    for f in FILES_TO_DELETE:
        s = delete_file(f, args.dry_run)
        if s:
            total_files += 1
            total_size  += s

    # Xoá data/ nếu không --keep-data
    if not args.keep_data:
        n, s = clear_directory(DATA_DIR, args.dry_run)
        total_files += n
        total_size  += s
    else:
        print(f"   ⏭  data/ — giữ lại (--keep-data)")

    print()
    if args.dry_run:
        print(f"🔍 [DRY RUN] Sẽ xoá: {total_files} files ({sizeof_fmt(total_size)})")
    else:
        print(f"✅ Đã xoá: {total_files} files ({sizeof_fmt(total_size)})")
        print("   Project đã được reset. Chạy lại notebook từ đầu.")
    print("=" * 55)


if __name__ == "__main__":
    main()


    # python cleanup.py
    # python cleanup.py --keep-data      # giữ lại data/, chỉ xoá features/models/reports
    # python cleanup.py --dry-run        # chỉ in ra, không xoá thật