from huggingface_hub import snapshot_download

for i in range(1):
    try:
        # 下载整个数据集（自动包含LFS大文件）
        snapshot_download(
            repo_id="JiayiChenPKU/BODex",
            repo_type="dataset",  # 指定是数据集
            local_dir="/media/mingrui/WD_BLACK/dataset/BODex",  # 本地保存路径
            resume_download=True,  # 支持断点续传
        )
    except:
        continue
