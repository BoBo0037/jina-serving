import requests

BASE_URL = "http://localhost:8888"

def test_endpoint(path, data, description):
    """测试端点"""
    print(f"\n🔍 测试 {description}")
    print(f"POST {path}")
    print(f"Data: {data}")

    response = requests.post(f"{BASE_URL}{path}", json=data)
    print(f"状态码: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        if "data" in result:
            # MaxKB 格式
            print(f"✅ 成功! Code: {result['code']}")
            print(f"   消息: {result['message']}")
            print(f"   形状: {result['data']['shape']}")
        else:
            # 原始格式
            print(f"✅ 成功!")
            print(f"   形状: {result['shape']}")
    else:
        print(f"❌ 失败: {response.text}")

def test_health():
    """测试健康检查"""
    print("\n🔍 测试健康检查")
    response = requests.get(f"{BASE_URL}/")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 服务状态: {result['code']} - {result['message']}")
    else:
        print(f"❌ 健康检查失败: {response.status_code}")

if __name__ == "__main__":
    print("🚀 开始测试 Jina V4 API")

    # 健康检查
    test_health()

    # 测试 MaxKB 格式
    test_endpoint("/encode/text", {
        "sentences": ["Hello world", "How are you?"],
        "task": "retrieval"
    }, "MaxKB 文本编码")

    # 测试原始格式
    test_endpoint("/encode/text", {
        "texts": ["Hello world", "How are you?"],
        "task": "retrieval"
    }, "原始文本编码")

    # 测试图像编码
    test_endpoint("/encode/image", {
        "image_urls": ["https://i.ibb.co/nQNGqL0/beach1.jpg"],
        "task": "retrieval"
    }, "MaxKB 图像编码")

    # 测试多模态
    test_endpoint("/encode/multimodal", {
        "texts": ["A beautiful beach"],
        "image_urls": ["https://i.ibb.co/nQNGqL0/beach1.jpg"],
        "task": "retrieval"
    }, "MaxKB 多模态编码")

    # 测试专用端点
    test_endpoint("/retrieval/query", ["Find information about AI"], "检索查询")
    test_endpoint("/text-matching", ["Hello", "World"], "文本匹配")

    print("\n✨ 测试完成!")