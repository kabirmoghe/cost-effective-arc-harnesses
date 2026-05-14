from shared.llm import create_async_client, get_default_model
import asyncio

from dotenv import load_dotenv

load_dotenv()

qwen_url = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
deepseek_url = "https://api.deepseek.com"

client = create_async_client(provider="deepseek")

async def get_response():
    response = await client.chat.completions.create(
        model='deepseek-chat',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        temperature=0)

    return response

if __name__ == "__main__": 
    res = asyncio.run(get_response())
    print(res.choices[0].message.content)
