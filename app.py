import streamlit as st
from core import RAGSystem
import time

# ページ設定
st.set_page_config(page_title="岩手県立大学 RAGチャットボット", page_icon="🎓")

st.title("🎓 岩手県立大学 AIアシスタント")
st.markdown("岩手県立大学に関する質問に、公式サイトの情報をもとに回答します。")

# RAGシステムの初期化（キャッシュを使用してモデルのロードを1回にする）
@st.cache_resource
def get_rag_system():
    rag = RAGSystem()
    rag.load_models()
    return rag

# データの準備（キャッシュを使用してスクレイピング/ベクトル化を1回にする）
@st.cache_data
def prepare_rag_data(_rag):
    # プログレスバーとステータス表示用のプレースホルダー
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def progress_callback(current, total):
        progress = current / total
        progress_bar.progress(progress)
        status_text.text(f"スクレイピング中: {current}/{total} ページ")
    
    _rag.prepare_data(progress_callback=progress_callback)
    
    progress_bar.empty()
    status_text.empty()
    return True

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# サイドバー
with st.sidebar:
    st.header("システム設定")
    st.info("💡 初回起動時は最大70ページのスクレイピングに数分かかります。")
    
    if st.button("キャッシュをクリアして再取得"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.divider()
    st.caption("🚀 Qwen2.5-0.5B (爆速版)")
    st.caption("📚 70ページ制限")

# システムのロード
try:
    with st.status("システムを起動中...", expanded=True) as status:
        st.write("モデルをロードしています...")
        rag = get_rag_system()
        st.write("データを準備しています（初回は数分かかります）...")
        prepare_rag_data(rag)
        status.update(label="準備完了！", state="complete", expanded=False)
except Exception as e:
    st.error(f"起動中にエラーが発生しました: {e}")
    st.stop()

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "elements" in message:
            with st.expander("📚 出典"):
                for url in message["elements"]:
                    st.write(f"- {url}")

# ユーザー入力
if prompt := st.chat_input("岩手県立大学について教えてください"):
    # ユーザーメッセージの表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # アシスタントの回答生成
    with st.chat_message("assistant"):
        with st.status("回答を作成しています...", expanded=True) as status:
            # 1. 検索
            st.write("🔍 関連資料を検索中...")
            context_texts, ref_urls = rag.search(prompt)
            combined_context = "\n\n".join(context_texts)
            
            # 2. 生成（ストリーミング）
            st.write("✍️ 回答を生成中...")
            status.update(label="回答中...", state="running", expanded=False)
        
        # 生成（同期）
        answer = rag.generate_answer(prompt, combined_context)
        
        # 回答の表示
        st.markdown(answer)
        
        # 出典の表示
        if ref_urls:
            with st.expander("📚 出典"):
                for url in ref_urls:
                    st.write(f"- {url}")
        
        # セッション履歴に保存
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "elements": ref_urls
        })
