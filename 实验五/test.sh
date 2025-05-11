#!/bin/bash

# 定义源目录和目标目录
source_dir="../data"
target_dir="../result"

# 检查data目录是否存在
if [ ! -d "$source_dir" ]; then
    echo "Error: Directory $source_dir does not exist."
    exit 1
fi

> log.txt

# 遍历data目录中的所有文件
for file in "$source_dir"/*/image_SRF_2/HR/*; do
    # 检查是否是文件（不是目录）
    if [ -f "$file" ]; then
        echo "Testing file: $file"
        echo -n "$file " >> log.txt
        python test.py --image-file "$file" >> log.txt
    fi
done

echo "All tests completed."

find "$source_dir" -type f \( -name "*_srcnn_*.png" -o -name "*_bicubic_*.png" \) | while read -r file; do
    # 获取文件相对路径（相对于 source_dir）
    relative_path="${file#$source_dir/}"
    
    # 获取目标目录路径（去掉文件名部分）
    target_subdir="$target_dir/$(dirname "$relative_path")"
    
    # 创建目标目录（如果不存在）
    mkdir -p "$target_subdir"
    
    # 移动文件
    mv "$file" "$target_subdir/"
    
done

echo "All result files have been moved."

python plot.py