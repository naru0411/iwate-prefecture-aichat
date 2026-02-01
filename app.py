import chainlit as cl
from core import RAGSystem
import asyncio

# RAGシステムの初期化
rag = RAGSystem()

@cl.on_chat_start
async def start():
    # 初回メッセージ
    msg = cl.Message(content="システムを起動中... (モデルのロードとデータの準備を行います)")
    await msg.send()
    
    # モデルのロードとデータの準備（重い処理なので非同期でラップして実行を検討するが、
    # 起動時に一度だけ必要なのでここで実行）
    try:
        # 他のスレッドで実行してUIをブロックしないようにする
        await asyncio.to_thread(rag.load_models)
        await asyncio.to_thread(rag.prepare_data)
        
        msg.content = "準備が完了しました！岩手県立大学について何でも聞いてください。"
        await msg.update()
    except Exception as e:
        msg.content = f"起動中にエラーが発生しました: {str(e)}"
        await msg.update()

@cl.on_message
async def main(message: cl.Message):
    # ステータス表示
    status_msg = cl.Message(content="")
    
    # 1. 検索フェーズ
    async with cl.Step(name="資料を検索中...") as step:
        context_texts, ref_urls = await asyncio.to_thread(rag.search, message.content)
        combined_context = "\n\n".join(context_texts)
        step.output = f"{len(ref_urls)} 個の関連資料が見つかりました。"

    # 2. 生成フェーズ
    async with cl.Step(name="回答を生成中...") as step:
        answer = await asyncio.to_thread(rag.generate_answer, message.content, combined_context)
        step.output = "回答の生成が完了しました。"

    # Sources (出典) の作成
    elements = []
    for i, url in enumerate(ref_urls):
        elements.append(
            cl.Text(name=f"Source {i+1}", content=url, display="inline")
        )

    # 最終回答の送信
    res_msg = cl.Message(content=answer, elements=elements)
    
    # 出典リストをテキストでも追加（Chainlit機能と併用）
    if ref_urls:
        res_msg.content += "\n\n**【参照リンク】**\n"
        for url in ref_urls:
            res_msg.content += f"- {url}\n"
            
    await res_msg.send()
