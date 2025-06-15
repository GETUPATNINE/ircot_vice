import json
import random
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from zhipuai import ZhipuAI

# 辅助函数
def content_to_text(content: str, max_num_words: int) -> str:
    content = " ".join(content.split(" ")[:max_num_words])
    content = (
        content.strip()
        if content.strip().startswith("Similar Question: ")
        else "Similar Question: " + content.strip()
    )
    return content

# 模型调用类
class ZhipuAIGenerator:
    def __init__(self, api_key="", model_name=""):
        """初始化ZhipuAI客户端"""
        print(f"Using model: {model_name}")
        self.client = ZhipuAI(api_key=api_key)
        self.model_name = model_name
        
    def generate_text_sequence(self, prompt, max_length=None):
        """使用ZhipuAI API生成文本"""
        # print("prompt:", prompt[:200] + "..." if len(prompt) > 200 else prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
            )
            
            # 提取生成的文本
            generated_text = response.choices[0].message.content
            
            # print(f"generated_text: {generated_text[:200]}..." if len(generated_text) > 200 else generated_text)
            return generated_text
            
        except Exception as e:
            print(f"API call error: {e}")
            return None

# 本地AQuA数据检索器
class AQuaRetriever:
    def __init__(
        self,
        retrieval_count=5,
        corpus_path="AQuA/train.tok.json",
        corpus_name="aqua_train",
    ):
        self.retrieval_count = retrieval_count
        self.corpus_name = corpus_name
        self.corpus_path = corpus_path
        
        # 加载训练数据
        print(f"Loading AQuA training data: {corpus_path}")
        self.train_data = []
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        self.train_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing training data: {e}")
            print(f"Successfully loaded {len(self.train_data)} training examples")
        except Exception as e:
            print(f"Failed to load training data: {e}")
            self.train_data = []
        
        # 准备向量化
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # 提取所有问题文本用于训练向量化器
        corpus = [item['question'] for item in self.train_data]
        if corpus:
            self.vectorizer.fit(corpus)
            self.question_vectors = self.vectorizer.transform(corpus)
            print("Question vector index created")
        else:
            print("Warning: No available training data, vector index not created")
            self.question_vectors = None
    
    def retrieve(self, query_text):
        """基于问题相似度，从训练数据中检索最相似的示例"""
        if not self.train_data or self.question_vectors is None:
            print("Warning: Training data not loaded or vector index not created, returning empty result")
            return [], []
        
        try:
            # 将查询转换为向量
            query_vector = self.vectorizer.transform([query_text])
            
            # 计算相似度
            similarities = cosine_similarity(query_vector, self.question_vectors)[0]
            
            # 获取最相似的k个问题的索引
            top_k_indices = similarities.argsort()[-self.retrieval_count:][::-1]
            
            titles = []
            contents = []
            
            # 检索相似问题及其答案
            for idx in top_k_indices:
                example = self.train_data[idx]
                question = example['question']
                options = example['options']
                rationale = example.get('rationale', "No explanation available")
                
                title = f"Question: {question}"
                options_text = "; ".join(options)

                content = (f"Question: {question}\n"
                           f"Options: {options_text}\n"
                           f"Explanation: {rationale}\n")
                
                titles.append(title)
                contents.append(content)
            
            # print(f"Retrieved {len(titles)} similar questions from training data")
            return titles, contents
            
        except Exception as e:
            print(f"Error retrieving similar questions: {e}")
            return [], []

# AQuA数据加载器 - 只用于加载样本问题
class AQuaLoader:
    def __init__(self, dataset_path):
        print(f"Loading AQuA dataset from {dataset_path}")
        self.data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
        print(f"Loaded {len(self.data)} problems from AQuA dataset")
    
    def get_problem(self, index=None):
        if index is None:
            index = random.randint(0, len(self.data) - 1)
        return self.data[index]
    
    def get_random_problem(self):
        return self.get_problem()

# 交替迭代 (Reason + Retrieve)
class IRCoTSystem:
    def __init__(
        self,
        initial_retrieval_count=5,
        iterative_retrieval_count=3,
        max_iterations=5,
        max_para_words=350,
        corpus_name="aqua_train",
        retriever=None,
        iterative_retriever=None,
        corpus_path="AQuA/train.tok.json",
        api_key="",
        model_name=""
    ):
        # 初始化本地知识检索器
        if retriever is None:
            self.retriever = AQuaRetriever(
                retrieval_count=initial_retrieval_count,
                corpus_name=corpus_name,
                corpus_path=corpus_path
            )
        else:
            self.retriever = retriever
        
        # 用于迭代检索的检索器
        if iterative_retriever is None:
            self.iterative_retriever = AQuaRetriever(
                retrieval_count=iterative_retrieval_count,
                corpus_name=corpus_name,
                corpus_path=corpus_path
            )
        else:
            self.iterative_retriever = iterative_retriever
        
        # 初始化ZhipuAI模型
        self.generator = ZhipuAIGenerator(api_key=api_key, model_name=model_name)
        
        self.max_iterations = max_iterations
        self.max_para_words = max_para_words

    def initial_retrieval(self, question):
        """执行初始检索，根据原始问题检索文档"""
        titles, contents = self.retriever.retrieve(question)
        return titles, contents
    
    def reason_step(self, question, options, contents, generated_sentences=[]):
        """生成下一步推理句子，整合之前的推理和搜索结果"""
        # 构建提示
        context = "\n".join([content_to_text(content, self.max_para_words) for content in contents])
        options_text = "; ".join(options)
        generation_so_far = " ".join(generated_sentences)
        
        # 根据是否有先前推理构建不同的提示
        if len(generated_sentences) > 0:
            prompt = (
                f"Search results: {context}\n"
                f"Question: {question}\n"
                f"Options: {options_text}\n"
                f"Previous reasoning: {generation_so_far}\n"
                f"You are an expert in mathematics, continue from the Previous reasoning with new insights. Be specific and show calculations when needed."
            )
        else:
            prompt = (
                f"Search results: {context}\n"
                f"Question: {question}\n"
                f"Options: {options_text}\n"
                f"As an expert in mathematics, analyze this problem step by step. Break down the calculations and show your reasoning clearly."
            )
        
        # 使用glm-z1-flash生成推理句子
        generated_texts = self.generator.generate_text_sequence(prompt, max_length=400)
        new_generation = generated_texts
        
        return new_generation
    
    def retrieve_step(self, cot_sentence, existing_titles=None, existing_contents=None):
        """使用CoT句子作为查询进行检索"""
        if existing_titles is None:
            existing_titles = []
        if existing_contents is None:
            existing_contents = []
        
        new_titles, new_contents = self.iterative_retriever.retrieve(cot_sentence)
        
        all_titles = existing_titles.copy()
        all_contents = existing_contents.copy()
        
        for title, content in zip(new_titles, new_contents):
            if title not in all_titles:
                all_titles.append(title)
                all_contents.append(content)
            else:
                # print(f"Warning: Duplicate title found: {title}")
                pass
                
        return all_titles, all_contents
    
    def run(self, question, options):
        """运行完整的IRCoT过程"""
        # 初始检索
        # print("--------- Start initial retrieval ---------")
        titles, contents = self.initial_retrieval(question)
        # print(f"Initial retrieval returned {len(titles)} documents")
        
        # 交替迭代
        generated_sentences = []
        
        # print("--------- Start alternating iteration ---------")
        for i in range(self.max_iterations):
            # print(f"\n--------- Iteration {i+1} ---------")
            
            # Reason步骤 - 生成下一个CoT句子
            new_sentence = self.reason_step(question, options, contents, generated_sentences)
            generated_sentences.append(new_sentence)
            
            # Retrieve步骤 - 使用新的CoT句子进行检索
            titles, contents = self.retrieve_step(new_sentence, titles, contents)
            # print(f"Got {len(titles)} documents in total")

        result = {
            "question": question,
            "options": options,
            "cot_reasoning": " ".join(generated_sentences),
            "retrieved_contents": "Here are some questions you can refer with:\n" + "\n".join(contents),
        }
        
        return result

# 使用示例
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--model_name", type=str, default="glm-z1-flash")
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--initial_retrieval_count", type=int, default=3)
    parser.add_argument("--iterative_retrieval_count", type=int, default=2)
    parser.add_argument("--corpus_path", type=str, default="AQuA/train.tok.json")
    parser.add_argument("--corpus_name", type=str, default="aqua_train")
    args = parser.parse_args()


    if args.api_key == "":
        print("----------------------------------------------------------------------")
        print("Warning: glm-z1-flash needs api key, you need to use your own api key!")
        print("----------------------------------------------------------------------")
        exit()
    
    print(f"Start running IRCoT system, using {args.model_name} model, retrieving similar questions from AQuA training data...")
    
    # 创建IRCoT系统
    ircot_system = IRCoTSystem(
        corpus_name=args.corpus_name,
        max_iterations=args.max_iterations,
        corpus_path=args.corpus_path,
        initial_retrieval_count=args.initial_retrieval_count,  # 初次检索的相似问题数量
        iterative_retrieval_count=args.iterative_retrieval_count,  # 迭代检索的相似问题数量
        api_key=args.api_key,
        model_name=args.model_name
    )
    
    # 从AQuA测试集加载一个问题
    aqua_loader = AQuaLoader("AQuA/test.tok.json")
    test_problem = aqua_loader.get_random_problem()
    
    question = test_problem["question"]
    options = test_problem["options"]
    correct = test_problem["correct"]
    
    print("========= Test Problem =========")
    print(f"Question: {question}")
    print(f"Options: {options}")
    print(f"Correct Answer: {correct}")
    
    # 运行IRCoT
    result = ircot_system.run(question, options)
    
    # 打印结果
    print("\n========= Final Result =========")
    print(f"Question: [{result['question']}]")
    print(f"Options: [{options}]")
    print(f"CoT Reasoning: [{result['cot_reasoning']}]")
    print(f"Retrieved Contents: [{result['retrieved_contents']}]")

if __name__ == "__main__":
    main()