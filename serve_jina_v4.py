import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, List, Optional
import torch
from transformers import AutoModel
from PIL import Image
import requests
from io import BytesIO
import base64

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    load_model()
    yield
    # 关闭时的清理工作（如果需要）

app = FastAPI(
    title="Jina Embeddings V4 API",
    version="1.0.0",
    lifespan=lifespan
)

model = None

# 响应模型
class BaseResponse(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None

class EmbeddingData(BaseModel):
    embeddings: List[List[float]]
    shape: List[int]
    task: str
    inputs: Optional[List[dict]] = None

def load_model():
    """加载模型"""
    global model
    if model is None:
        print("Loading Jina Embeddings V4 model...")
        model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v4",
            trust_remote_code=True,
            dtype=torch.bfloat16
        )
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        print(f"Model loaded on device: {device}")

def load_image(image_input: str) -> Image.Image:
    """加载图片"""
    try:
        if image_input.startswith(('http://', 'https://')):
            response = requests.get(image_input)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        elif image_input.startswith('data:image'):
            image_data = base64.b64decode(image_input.split(',')[1])
            return Image.open(BytesIO(image_data))
        else:
            return Image.open(image_input)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")

def check_model():
    """检查模型是否已加载"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型正在加载中，请稍后再试")

def create_embeddings_response(embeddings, task="retrieval", maxkb_format=False):
    """创建响应"""
    embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

    if maxkb_format:
        data = EmbeddingData(
            embeddings=embeddings_list,
            shape=list(embeddings.shape) if hasattr(embeddings, 'shape') else [len(embeddings_list)],
            task=task
        )
        return BaseResponse(
            code=200,
            message=f"成功生成 {len(embeddings_list)} 个向量",
            data=data
        )
    else:
        return {
            "embeddings": embeddings_list,
            "shape": list(embeddings.shape) if hasattr(embeddings, 'shape') else [len(embeddings_list)]
        }

@app.get("/", response_model=BaseResponse)
async def root():
    """健康检查"""
    status = "healthy" if model else "loading"
    message = "Jina Embeddings V4 API 服务运行正常" if model else "模型正在加载中，请稍后再试"
    return BaseResponse(
        code=200 if model else 503,
        message=message,
        data={"status": status, "model_loaded": bool(model), "service": "Jina Embeddings V4 API"}
    )

@app.post("/encode/text")
async def encode_text(request: dict):
    """文本编码 - 支持 MaxKB 和原始格式"""
    check_model()

    # 判断格式
    if "sentences" in request:
        texts = request["sentences"]
        task = request.get("task", "retrieval")
        prompt_name = request.get("prompt_name")
        maxkb_format = True
    elif "texts" in request:
        texts = request["texts"]
        task = request.get("task", "retrieval")
        prompt_name = request.get("prompt_name")
        maxkb_format = False
    else:
        return BaseResponse(code=400, message="请求必须包含 'texts' 或 'sentences' 字段")

    if not texts:
        return BaseResponse(code=400, message="文本列表不能为空")

    try:
        embeddings = model.encode_text(
            texts=texts,
            task=task,
            prompt_name=prompt_name
        )
        return create_embeddings_response(embeddings, task, maxkb_format)
    except Exception as e:
        return BaseResponse(code=500, message=f"文本编码失败: {str(e)}")

@app.post("/encode/image")
async def encode_image(request: dict):
    """图像编码 - 支持 MaxKB 和原始格式"""
    check_model()

    # 判断格式
    if "image_urls" in request:
        image_urls = request["image_urls"]
        task = request.get("task", "retrieval")
        maxkb_format = True
    elif "images" in request:
        image_urls = request["images"]
        task = request.get("task", "retrieval")
        maxkb_format = False
    else:
        return BaseResponse(code=400, message="请求必须包含 'images' 或 'image_urls' 字段")

    if not image_urls:
        return BaseResponse(code=400, message="图像URL列表不能为空")

    try:
        images = [load_image(url) for url in image_urls]
        embeddings = model.encode_image(images=images, task=task)
        return create_embeddings_response(embeddings, task, maxkb_format)
    except Exception as e:
        return BaseResponse(code=500, message=f"图像编码失败: {str(e)}")

@app.post("/encode/multimodal")
async def encode_multimodal(request: dict):
    """多模态编码 - 支持 MaxKB 和原始格式"""
    check_model()

    # 判断格式并提取数据
    if "image_urls" in request:
        texts = request.get("texts")
        image_urls = request.get("image_urls")
        task = request.get("task", "retrieval")
        prompt_name = request.get("prompt_name")
        maxkb_format = True
    else:
        texts = request.get("texts")
        image_urls = request.get("images")
        task = request.get("task", "retrieval")
        maxkb_format = False

    if not texts and not image_urls:
        return BaseResponse(code=400, message="文本和图像不能同时为空")

    try:
        all_embeddings = []

        # 编码文本
        if texts:
            text_embeddings = model.encode_text(texts=texts, task=task, prompt_name=prompt_name)
            text_list = text_embeddings.tolist() if hasattr(text_embeddings, 'tolist') else text_embeddings
            all_embeddings.extend(text_list)

        # 编码图像
        if image_urls:
            images = [load_image(url) for url in image_urls]
            image_embeddings = model.encode_image(images=images, task=task)
            image_list = image_embeddings.tolist() if hasattr(image_embeddings, 'tolist') else image_embeddings
            all_embeddings.extend(image_list)

        if maxkb_format:
            input_descriptors = []
            position = 0

            if texts:
                for i in range(len(texts)):
                    input_descriptors.append({"kind": "text", "original_index": i, "position": position})
                    position += 1

            if image_urls:
                for i in range(len(image_urls)):
                    input_descriptors.append({"kind": "image", "original_index": i, "position": position})
                    position += 1

            data = EmbeddingData(
                embeddings=all_embeddings,
                shape=[len(all_embeddings), len(all_embeddings[0]) if all_embeddings else 0],
                task=task,
                inputs=input_descriptors if input_descriptors else None
            )
            return BaseResponse(
                code=200,
                message=f"成功生成 {len(all_embeddings)} 个多模态向量",
                data=data
            )
        else:
            return {
                "embeddings": all_embeddings,
                "shape": [len(all_embeddings), len(all_embeddings[0]) if all_embeddings else 0]
            }

    except Exception as e:
        return BaseResponse(code=500, message=f"多模态编码失败: {str(e)}")

# MaxKB 专用端点
@app.post("/retrieval/query", response_model=BaseResponse)
async def retrieval_query(sentences: List[str]):
    """检索查询"""
    return await encode_text({"sentences": sentences, "task": "retrieval", "prompt_name": "query"})

@app.post("/retrieval/passage", response_model=BaseResponse)
async def retrieval_passage(sentences: List[str]):
    """检索段落"""
    return await encode_text({"sentences": sentences, "task": "retrieval", "prompt_name": "passage"})

@app.post("/text-matching", response_model=BaseResponse)
async def text_matching(sentences: List[str]):
    """文本匹配"""
    return await encode_text({"sentences": sentences, "task": "text-matching"})

@app.post("/code/query", response_model=BaseResponse)
async def code_query(sentences: List[str]):
    """代码查询"""
    return await encode_text({"sentences": sentences, "task": "code", "prompt_name": "query"})

@app.post("/code/passage", response_model=BaseResponse)
async def code_passage(sentences: List[str]):
    """代码段落"""
    return await encode_text({"sentences": sentences, "task": "code", "prompt_name": "passage"})

if __name__ == "__main__":
    import uvicorn
    print("Starting Jina V4 Embeddings API server...")
    print("API Documentation: http://localhost:8888/docs")
    print("Health Check: http://localhost:8888/health")
    uvicorn.run("serve_jina_v4:app", host="0.0.0.0", port=8888, reload=False, workers=1)