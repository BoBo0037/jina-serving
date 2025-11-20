#!/bin/bash

echo "🚀 启动 Jina V4 Embeddings API 服务..."
echo "📍 服务地址: http://localhost:8888"
echo "📖 API 文档: http://localhost:8888/docs"
echo "💚 健康检查: http://localhost:8888/health"
echo ""

# 检查依赖是否已安装
echo "🔍 检查依赖..."
python -c "import fastapi, uvicorn, transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 安装依赖..."
    pip install -r requirements.txt
else
    echo "✅ 依赖已就绪"
fi

echo "🎯 启动服务器..."
echo "⚡ 按 Ctrl+C 停止服务"
echo ""
python serve_jina_v4.py