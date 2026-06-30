import os
import shutil
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pathspec

SOURCE_DIR = os.path.abspath(".")
DEST_DIR = os.path.abspath("G:/My Drive/tBTDD")
IGNORE_FILE = "drive.ignore"

def load_ignore_spec():
    """Đọc file drive.ignore và lập bộ lọc chắn."""
    patterns = ['.git/', '.vscode/', '__pycache__/', '*.pyc', IGNORE_FILE, 'sync.py']
    if os.path.exists(IGNORE_FILE):
        with open(IGNORE_FILE, 'r', encoding='utf-8') as f:
            patterns.extend([line.strip() for line in f if line.strip() and not line.startswith('#')])
    return pathspec.PathSpec.from_lines('gitwildmatch', patterns)

def needs_sync(src_path, dest_path):
    """
    Kiểm tra xem file có thực sự cần đồng bộ không.
    Tránh copy thừa do Google Drive làm lệch timestamp hoặc VSCode lưu đè nhiều lần.
    """
    if not os.path.exists(dest_path):
        return True
    
    # 1. So sánh kích thước file (nhanh và chính xác nếu nội dung thay đổi)
    if os.path.getsize(src_path) != os.path.getsize(dest_path):
        return True
        
    # 2. So sánh thời gian. Thêm dung sai 2 giây vì Google Drive thường làm xê dịch mtime
    src_mtime = os.path.getmtime(src_path)
    dest_mtime = os.path.getmtime(dest_path)
    
    if src_mtime > dest_mtime + 2.0: 
        return True
        
    return False

class SyncHandler(FileSystemEventHandler):
    def __init__(self, spec):
        self.spec = spec

    def is_ignored(self, path):
        rel_path = os.path.relpath(path, SOURCE_DIR).replace('\\', '/')
        # Bắt dính cả vỏ thư mục lẫn file bên trong
        return self.spec.match_file(rel_path) or self.spec.match_file(rel_path + '/') or rel_path == '.'

    def sync_file(self, src_path):
        # Thêm check exists để tránh lỗi khi VSCode xóa file tạm (swp file) quá nhanh
        if not os.path.exists(src_path) or self.is_ignored(src_path) or os.path.isdir(src_path): 
            return
        
        rel_path = os.path.relpath(src_path, SOURCE_DIR)
        dest_path = os.path.join(DEST_DIR, rel_path)
        
        if needs_sync(src_path, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            try:
                shutil.copy2(src_path, dest_path)
                print(f"Đã up: {rel_path}")
            except Exception as e:
                print(f"Lỗi khi up {rel_path}: {e}")

    def delete_file(self, src_path):
        if self.is_ignored(src_path): 
            return
        
        rel_path = os.path.relpath(src_path, SOURCE_DIR)
        dest_path = os.path.join(DEST_DIR, rel_path)
        
        try:
            if os.path.exists(dest_path):
                if os.path.isdir(dest_path):
                    shutil.rmtree(dest_path)
                else:
                    os.remove(dest_path)
                print(f"Đã xóa: {rel_path}")
        except Exception as e:
            print(f"Lỗi khi xóa {rel_path}: {e}")

    # Bắt các sự kiện lưu, tạo mới, xóa, đổi tên
    def on_modified(self, event): self.sync_file(event.src_path)
    def on_created(self, event): self.sync_file(event.src_path)
    def on_deleted(self, event): self.delete_file(event.src_path)
    def on_moved(self, event):
        self.delete_file(event.src_path)
        self.sync_file(event.dest_path)

def initial_sync(spec):
    print("Đang quét để đồng bộ lần đầu (Chỉ up file mới/có thay đổi)...")
    sync_count = 0
    for root, dirs, files in os.walk(SOURCE_DIR):
        rel_root = os.path.relpath(root, SOURCE_DIR).replace('\\', '/')
        
        # Cắt đứt đường đi vào các thư mục bị cấm ngay từ đầu
        dirs[:] = [d for d in dirs if not spec.match_file(os.path.join(rel_root, d).replace('\\', '/') + '/')]
        
        for file in files:
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, SOURCE_DIR).replace('\\', '/')
            
            if not spec.match_file(rel_path):
                dest_path = os.path.join(DEST_DIR, rel_path)
                
                # Áp dụng hàm needs_sync thay vì kiểm tra mtime thô
                if needs_sync(src_path, dest_path):
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    try:
                        shutil.copy2(src_path, dest_path)
                        print(f"Đã đồng bộ: {rel_path}")
                        sync_count += 1
                    except Exception as e:
                        print(f"Lỗi khi đồng bộ {rel_path}: {e}")
                        
    if sync_count == 0:
        print("Tất cả các file đều đã được cập nhật mới nhất.")

if __name__ == "__main__":
    spec = load_ignore_spec()
    initial_sync(spec)
    
    event_handler = SyncHandler(spec)
    observer = Observer()
    observer.schedule(event_handler, SOURCE_DIR, recursive=True)
    observer.start()
    
    print(f"\nĐang theo dõi thay đổi... (Bấm Ctrl+C để dừng)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()