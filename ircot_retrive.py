import re
import json
import time
from typing import List, Dict, Any, Optional
import requests
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from functools import lru_cache
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 辅助函数
def safe_post_request(url, params):
    for _ in range(10):
        try:
            return requests.post(url, json=params)
        except:
            print("Post request didn't succeed. Will wait 20s and retry.")
            time.sleep(20)
    raise Exception("Post request couldn't succeed after several attempts.")

def remove_wh_words(text: str) -> str:
    wh_words = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}
    words = [word for word in text.split(" ") if word.strip().lower() not in wh_words]
    text = " ".join(words)
    return text

def is_reasoning_sentence(sentence: str) -> bool:
    starters = ["thus ", "thus,", "so ", "so,", "that is,", "therefore", "hence"]
    for starter in starters:
        if sentence.lower().startswith(starter):
            return True
    regex = re.compile("(.*)(\d[\d,]*\.?\d+|\d+) ([+-]) (\d[\d,]*\.?\d+|\d+) = (\d[\d,]*\.?\d+|\d+)(.*)")
    match = bool(re.match(regex, sentence))
    return match

def remove_reasoning_sentences(sentences: List[str]) -> List[str]:
    return [sentence for sentence in sentences if not is_reasoning_sentence(sentence)]

def para_to_text(title: str, para: str, max_num_words: int) -> str:
    para = " ".join(para.split(" ")[:max_num_words])
    para = (
        para.strip()
        if para.strip().startswith("Wikipedia Title: ")
        else "Wikipedia Title: " + title + "\n" + para.strip()
    )
    return para

# 2. 模型加载类
class FlanT5Generator:
    def __init__(self, model_name="google/flan-t5-small", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Flan-T5 model on {self.device}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
    def generate_text_sequence(self, prompt, max_length=100, num_return_sequences=1):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs, 
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=0.7,
        )
        
        results = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            results.append((text, 1.0))  # Score is placeholder
        
        return results

# 3. 外部知识检索器 (External Wikipedia Retriever)
class BM25Retriever:
    def __init__(
        self,
        retriever_host,
        retriever_port,
        retrieval_count=5,
        corpus_name="wikipedia",
        document_type="paragraph_text",
        global_max_num_paras=15,
    ):
        self.retriever_host = retriever_host
        self.retriever_port = retriever_port
        self.retrieval_count = retrieval_count
        self.corpus_name = corpus_name
        self.document_type = document_type
        self.global_max_num_paras = global_max_num_paras
    
    def retrieve(self, query_text):
        query_text = remove_wh_words(query_text)
        
        params = {
            "retrieval_method": "retrieve_from_elasticsearch",
            "query_text": query_text,
            "max_hits_count": self.retrieval_count,
            "corpus_name": self.corpus_name,
            "document_type": self.document_type,
        }
        
        url = self.retriever_host.rstrip("/") + ":" + str(self.retriever_port) + "/retrieve"
        result = safe_post_request(url, params)
        
        titles = []
        paras = []
        
        if result.ok:
            result = result.json()
            retrieval = result["retrieval"]
            
            for item in retrieval:
                if item["corpus_name"] != self.corpus_name:
                    continue
                    
                title = item["title"]
                para = item.get("paragraph_text", "")
                
                if len(para.split(" ")) > 600:
                    print("WARNING: Discarding a retrieved paragraph as it's excessively long.")
                    continue
                    
                if title not in titles and len(paras) < self.global_max_num_paras:
                    titles.append(title)
                    paras.append(para)
        
        return titles, paras

# 4. AQuA数据加载器 - 只用于加载样本问题
class AQuaLoader:
    def __init__(self, dataset_path):
        print(f"Loading AQuA dataset from {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} problems from AQuA dataset")
    
    def get_problem(self, index=None):
        if index is None:
            index = random.randint(0, len(self.data) - 1)
        return self.data[index]
    
    def get_random_problem(self):
        return self.get_problem()

# 5. 交替迭代 (Reason + Retrieve)
class IRCoTSystem:
    def __init__(
        self,
        retriever_host,
        retriever_port,
        initial_retrieval_count=5,
        iterative_retrieval_count=3,
        max_iterations=5,
        max_para_words=350,
        corpus_name="wikipedia",
        answer_extractor_regex="[Tt]he answer is ([A-E])\\.?",
    ):
        # 初始化外部知识检索器
        self.retriever = BM25Retriever(
            retriever_host=retriever_host,
            retriever_port=retriever_port,
            retrieval_count=initial_retrieval_count,
            corpus_name=corpus_name
        )
        
        # 用于迭代检索的检索器
        self.iterative_retriever = BM25Retriever(
            retriever_host=retriever_host,
            retriever_port=retriever_port,
            retrieval_count=iterative_retrieval_count,
            corpus_name=corpus_name
        )
        
        # 初始化Flan-T5模型
        self.generator = FlanT5Generator(model_name="google/flan-t5-large")
        
        self.max_iterations = max_iterations
        self.max_para_words = max_para_words
        self.answer_extractor_regex = re.compile(answer_extractor_regex)
        
    def initial_retrieval(self, question, options):
        """执行初始检索，根据原始问题检索文档"""
        # 构建更丰富的查询，包含问题中的关键术语
        query = question + " " + " ".join(options)
        titles, paras = self.retriever.retrieve(query)
        return titles, paras
    
    def reason_step(self, question, options, titles, paras, generated_sentences=None):
        """生成下一步推理句子"""
        if generated_sentences is None:
            generated_sentences = []
        
        # 构建提示
        context = "\n\n".join([para_to_text(title, para, self.max_para_words) for title, para in zip(titles, paras)])
        options_text = "\n".join(options)
        generation_so_far = " ".join(generated_sentences)
        
        prompt = f"Context: {context}\n\nQuestion: {question}\nOptions:\n{options_text}\n\nReasoning: {generation_so_far}"
        
        # 使用Flan-T5生成推理句子
        output_text_scores = self.generator.generate_text_sequence(prompt, max_length=150)
        new_generation = output_text_scores[0][0].strip()
        
        # 简单分句处理
        if "." in new_generation:
            sentences = new_generation.split(".")
            new_generation = sentences[0] + "."
            
        return new_generation
    
    def retrieve_step(self, cot_sentence, existing_titles=None, existing_paras=None):
        """使用CoT句子作为查询进行检索"""
        if existing_titles is None:
            existing_titles = []
        if existing_paras is None:
            existing_paras = []
            
        new_titles, new_paras = self.iterative_retriever.retrieve(cot_sentence)
        
        # 合并并去重文档
        all_titles = existing_titles.copy()
        all_paras = existing_paras.copy()
        
        for title, para in zip(new_titles, new_paras):
            if title not in all_titles:
                all_titles.append(title)
                all_paras.append(para)
                
        return all_titles, all_paras
    
    def extract_answer(self, cot_sentences):
        """从CoT中提取最终答案（选项A-E）"""
        full_cot = " ".join(cot_sentences)
        match = self.answer_extractor_regex.search(full_cot)
        if match:
            return match.group(1)
        
        # 如果没有匹配正则表达式，查找选项A-E的直接引用
        for option in ["A", "B", "C", "D", "E"]:
            if f"answer is {option}" in full_cot.lower() or f"answer: {option}" in full_cot.lower():
                return option
        
        return None
    
    def run(self, question, options):
        """运行完整的IRCoT过程"""
        # 1. 初始检索
        titles, paras = self.initial_retrieval(question, options)
        print(f"Initial retrieval returned {len(titles)} documents")
        
        # 2. 交替迭代
        generated_sentences = []
        answer = None
        
        for i in range(self.max_iterations):
            print(f"\nIteration {i+1}:")
            
            # Reason步骤 - 生成下一个CoT句子
            new_sentence = self.reason_step(question, options, titles, paras, generated_sentences)
            generated_sentences.append(new_sentence)
            print(f"Generated reasoning: {new_sentence}")
            
            # 检查是否已生成答案
            potential_answer = self.extract_answer(generated_sentences)
            if potential_answer:
                answer = potential_answer
                print(f"Found answer: {answer}")
                break
                
            # Retrieve步骤 - 使用新的CoT句子检索
            titles, paras = self.retrieve_step(new_sentence, titles, paras)
            print(f"Retrieved {len(titles)} documents in total")
        
        # 如果没有找到答案，尝试从所有生成的CoT中提取
        if not answer:
            answer = self.extract_answer(generated_sentences)
            
        result = {
            "question": question,
            "options": options,
            "cot_reasoning": generated_sentences,
            "retrieved_titles": titles,
            "answer": answer
        }
        
        return result

# 6. 使用示例
def main():
    # 配置参数
    retriever_host = "http://localhost"  # 替换为实际的检索服务器地址
    retriever_port = "8000"              # 替换为实际的检索服务器端口
    corpus_name = "wikipedia"            # 使用Wikipedia作为知识来源
    
    # 创建IRCoT系统
    ircot_system = IRCoTSystem(
        retriever_host=retriever_host,
        retriever_port=retriever_port,
        corpus_name=corpus_name,
        max_iterations=5
    )
    
    # 从AQuA测试集加载一个问题
    aqua_loader = AQuaLoader("AQuA/test.json")
    test_problem = aqua_loader.get_random_problem()
    
    question = test_problem["question"]
    options = test_problem["options"]
    correct = test_problem["correct"]
    
    print("===== Test Problem =====")
    print(f"Question: {question}")
    print(f"Options: {options}")
    print(f"Correct Answer: {correct}")
    
    # 运行IRCoT
    result = ircot_system.run(question, options)
    
    # 打印结果
    print("\n===== Final Result =====")
    print(f"Question: {result['question']}")
    print(f"CoT Reasoning: {' '.join(result['cot_reasoning'])}")
    print(f"Answer: {result['answer']}")
    print(f"Correct Answer: {correct}")
    print(f"Retrieved {len(result['retrieved_titles'])} documents")
    print(f"Result is correct: {result['answer'] == correct}")

if __name__ == "__main__":
    main()