"""API端点测试"""
import os

def test_health_endpoint(test_client):
    """测试健康检查端点"""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_files_list_endpoint(test_client, test_files, test_dirs):
    """测试文件列表端点"""
    response = test_client.get("/api/files/list")
    assert response.status_code == 200
    data = response.json()
    assert "files" in data
    assert isinstance(data["files"], list)

def test_files_recent_endpoint(test_client, test_files):
    """测试最近文件端点"""
    response = test_client.get("/api/files/recent")
    assert response.status_code == 200
    data = response.json()
    assert "files" in data

def test_files_stats_endpoint(test_client, test_files):
    """测试文件统计端点"""
    response = test_client.get("/api/files/stats")
    assert response.status_code == 200
    data = response.json()
    assert "output_dir" in data
    assert "total_files" in data["output_dir"]
    assert "total_size_mb" in data["output_dir"]

def test_files_delete_single(test_client, test_files):
    """测试单个文件删除"""
    response = test_client.post("/api/files/delete", data={"filename": "source_file.mp3"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

def test_files_delete_not_found(test_client):
    """测试删除不存在的文件"""
    response = test_client.post("/api/files/delete", data={"filename": "nonexistent_file.mp3"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False

def test_files_delete_batch(test_client, test_files):
    """测试批量删除文件"""
    files_to_delete = ["task_1234567890_test_temp.mp3", "processed_file.mp3"]
    
    response = test_client.post("/api/files/delete_batch", data={"filenames": ",".join(files_to_delete)})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "deleted_count" in data
    assert "deleted_files" in data

def test_files_cleanup(test_client, test_files):
    """测试清理临时文件"""
    response = test_client.post("/api/files/cleanup")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "cleaned_count" in data
    assert "cleaned_size_mb" in data

def test_files_browse_endpoint(test_client):
    """测试目录浏览端点"""
    response = test_client.get("/api/files/browse")
    assert response.status_code == 200
    data = response.json()
    assert "directories" in data
    assert "files" in data

def test_files_download_endpoint(test_client, test_files):
    """测试文件下载端点"""
    response = test_client.get("/api/files/download/processed_file.mp3")
    assert response.status_code == 200 or response.status_code == 404

def test_audio_waveform_endpoint(test_client, sample_audio_file):
    """测试波形图端点"""
    response = test_client.post("/api/audio/waveform", data={"input_path": sample_audio_file})
    assert response.status_code == 200
    data = response.json()
    assert "success" in data

def test_progress_endpoint(test_client):
    """测试进度查询端点"""
    response = test_client.get("/api/media/progress/nonexistent_task")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_result_endpoint(test_client):
    """测试结果查询端点"""
    response = test_client.get("/api/media/result/nonexistent_task")
    assert response.status_code == 200 or response.status_code == 404