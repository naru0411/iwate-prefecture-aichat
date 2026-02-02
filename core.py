import re
import time
import requests
import numpy as np
import torch
import pickle
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from markdownify import markdownify as md
from typing import List, Dict, Tuple, Generator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import urllib3
from threading import Thread

# SSL証明書エラーの警告を非表示
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class RAGSystem:
    def __init__(self, cache_path="data_full.pkl"):
        self.cache_path = cache_path
        self.tokenizer = None
        self.model = None
        self.embd_model = None
        self.chunked_data = []
        self.chunked_metadata = []
        self.docs_embeddings = None
        
        self.gen_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.embd_model_name = "intfloat/multilingual-e5-small"

    def load_models(self):
        print("Loading models with 4-bit quantization...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.gen_model_name, trust_remote_code=True)
        
        # 4-bit量子化設定（Windows互換性を考慮）
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.gen_model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            print("Model loaded with 4-bit quantization successfully.")
        except Exception as e:
            print(f"Quantization failed ({e}), loading without quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.gen_model_name,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            )
        
        self.embd_model = SentenceTransformer(self.embd_model_name, trust_remote_code=True, device="cuda" if torch.cuda.is_available() else "cpu")

    def fetch_ipu_pages_clean(self, base_url: str="https://www.iwate-pu.ac.jp/", max_pages: int=133, progress_callback=None):
        results = []
        visited_urls = set()
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

        with requests.Session() as s:
            s.headers.update(headers)
            target_links = []
            important_urls = [
                "https://www.iwate-pu.ac.jp/faculty/",
                "https://www.iwate-pu.ac.jp/examination/all.html"
            ]
            for imp_url in important_urls:
                target_links.append({"url": imp_url, "title": "重要ページ"})
                visited_urls.add(imp_url)

            try:
                response = s.get(base_url, timeout=30, verify=False)
                response.encoding = response.apparent_encoding
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a", href=True):
                    full_url = urljoin(base_url, link["href"])
                    if ("iwate-pu.ac.jp" in full_url) and \
                       (full_url not in visited_urls) and \
                       (not full_url.endswith((".jpg", ".png", ".pdf", ".zip", ".css", ".js"))) and \
                       ("#" not in full_url):
                        target_links.append({"url": full_url, "title": link.text.strip()})
                        visited_urls.add(full_url)
            except Exception: pass

            total_links = min(len(target_links), max_pages)
            for i, target in enumerate(target_links):
                if i >= max_pages: break
                
                # 進捗コールバック
                if progress_callback:
                    progress_callback(i + 1, total_links)
                
                try:
                    res = s.get(target["url"], timeout=30, verify=False)
                    res.encoding = res.apparent_encoding
                    soup = BeautifulSoup(res.text, "html.parser")
                    title = soup.title.text.strip() if soup.title else target["title"]

                    noise_selectors = ['header', 'footer', 'nav', 'noscript', 'script', 'style', '.topic_path', '.pankuzu', '#pan', '.side_menu', '.search_area']
                    for selector in noise_selectors:
                        for noise in soup.select(selector): noise.decompose()

                    main_content = soup.find(id="contents") or soup.find("main") or soup.body
                    if main_content:
                        raw_text = md(str(main_content), strip=['a', 'img', 'script', 'style', 'iframe'])
                        lines = raw_text.split('\n')
                        cleaned_lines = [line.strip() for line in lines if line.strip() and line.strip() not in ["ホーム", "Home", "TOP"] and not line.startswith(">") and line.strip() != "学部・大学院等" and not re.match(r'^[-=]{3,}$', line.strip())]
                        content_text = "\n".join(cleaned_lines)
                        content_text = re.sub(r'\n{3,}', '\n\n', content_text)
                        if len(content_text) > 50:
                            results.append({"title": title, "url": target["url"], "content": content_text})
                            time.sleep(0.5)
                except Exception: pass
        return results

    def token_based_chunking(self, search_results, chunk_size=500):
        chunks = []
        metadatas = []
        for res in search_results:
            title, url, content = res["title"], res["url"], res["content"]
            source_str = f" (出典: {title})"
            source_tokens = len(self.tokenizer.encode(source_str, add_special_tokens=False))
            effective_limit = chunk_size - source_tokens
            sentences = content.replace("。", "。\n").split("\n")
            current_chunk_text = ""
            current_chunk_tokens = 0
            for sentence in sentences:
                if not sentence.strip(): continue
                sent_token_len = len(self.tokenizer.encode(sentence, add_special_tokens=False))
                if current_chunk_tokens + sent_token_len > effective_limit:
                    if current_chunk_text:
                        chunks.append(current_chunk_text + source_str)
                        metadatas.append({"title": title, "url": url})
                    current_chunk_text = sentence
                    current_chunk_tokens = sent_token_len
                else:
                    current_chunk_text += sentence
                    current_chunk_tokens += sent_token_len
            if current_chunk_text:
                chunks.append(current_chunk_text + source_str)
                metadatas.append({"title": title, "url": url})
        return chunks, metadatas

    def prepare_data(self, force_refresh=False, progress_callback=None):
        if not force_refresh and os.path.exists(self.cache_path):
            print(f"Loading cached data from {self.cache_path}...")
            with open(self.cache_path, "rb") as f:
                cache = pickle.load(f)
                self.chunked_data = cache["data"]
                self.chunked_metadata = cache["metadata"]
                self.docs_embeddings = cache["embeddings"]
            return

        print("No cache found. Starting full scraping (133 pages)...")
        raw_data = self.fetch_ipu_pages_clean(max_pages=133, progress_callback=progress_callback)
        self.chunked_data, self.chunked_metadata = self.token_based_chunking(raw_data)
        
        print("Vectorizing data with multilingual-e5-small...")
        # e5 models usually require 'passage: ' prefix for documents
        passage_prefixed_data = [f"passage: {chunk}" for chunk in self.chunked_data]
        self.docs_embeddings = self.embd_model.encode(passage_prefixed_data, batch_size=32, show_progress_bar=False)
        norms = np.linalg.norm(self.docs_embeddings, axis=1, keepdims=True)
        self.docs_embeddings = self.docs_embeddings / np.clip(norms, 1e-12, None)

        with open(self.cache_path, "wb") as f:
            pickle.dump({
                "data": self.chunked_data,
                "metadata": self.chunked_metadata,
                "embeddings": self.docs_embeddings
            }, f)
        print("Data preparation complete and cached.")

    def search(self, query: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
        # e5 models require 'query: ' prefix for queries
        query_vec = self.embd_model.encode([f"query: {query}"], show_progress_bar=False)[0]
        query_vec = query_vec / np.linalg.norm(query_vec)
        raw_scores = cosine_similarity([query_vec], self.docs_embeddings)[0]

        adjusted_scores = raw_scores.copy()
        BOOST_URL_KEYWORD = ("faculty", "examination/all.html", "access")
        BOOST_FACTOR = 1.3
        for i, meta in enumerate(self.chunked_metadata):
            if any(keyword in meta['url'] for keyword in BOOST_URL_KEYWORD):
                adjusted_scores[i] *= BOOST_FACTOR

        top_indices = adjusted_scores.argsort()[::-1][:top_k]
        context_texts = []
        ref_urls = []
        seen_urls = set()
        for idx in top_indices:
            context_texts.append(f"【資料】(出典:{self.chunked_metadata[idx]['title']})\n{self.chunked_data[idx]}")
            url = self.chunked_metadata[idx]['url']
            if url not in seen_urls:
                ref_urls.append(url)
                seen_urls.add(url)
        return context_texts, ref_urls

    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"""あなたは岩手県立大学の広報アシスタントAIです。
以下の【参照資料】の内容を統合して、ユーザーの【質問】に詳しく答えてください。

### 重要ルール
1. 複数の資料を組み合わせて、可能な限り回答を作成してください。
2. 回答文の中にURLは含めないでください。嘘も絶対に書かないでください。
3. **どうしても情報が見つからない場合は、言い訳をせずに『NO_INFO』とだけ出力してください。余計な文章は不要です。**

### 質問
{query}

### 参照資料
{context}

### 回答
"""
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.0, do_sample=False)
        
        answer = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        if "NO_INFO" in answer or "申し訳ありません" in answer:
            return "申し訳ありません。現時点の資料には詳しい記載がありませんでした。\n回答に近いと思われる以下のWebページをご確認いただけますでしょうか。"
        return answer

    def generate_answer_stream(self, query: str, context: str) -> Generator[str, None, None]:
        """ストリーミング形式で回答を生成"""
        prompt = f"""あなたは岩手県立大学の広報アシスタントAIです。
以下の【参照資料】の内容を統合して、ユーザーの【質問】に詳しく答えてください。

### 重要ルール
1. 複数の資料を組み合わせて、可能な限り回答を作成してください。
2. 回答文の中にURLは含めないでください。嘘も絶対に書かないでください。
3. **どうしても情報が見つからない場合は、言い訳をせずに『NO_INFO』とだけ出力してください。余計な文章は不要です。**

### 質問
{query}

### 参照資料
{context}

### 回答
"""
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        # ストリーマーの設定
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1024,
            temperature=0.0,
            do_sample=False
        )

        # 別スレッドで生成を開始
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # ストリーミング出力
        full_response = ""
        for new_text in streamer:
            full_response += new_text
            yield new_text

        thread.join()

        # NO_INFO チェック
        if "NO_INFO" in full_response or "申し訳ありません" in full_response:
            yield "\n\n申し訳ありません。現時点の資料には詳しい記載がありませんでした。\n回答に近いと思われる以下のWebページをご確認いただけますでしょうか。"
