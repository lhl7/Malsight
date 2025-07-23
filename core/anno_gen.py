import json
import httpx
import requests
import base64
import os
import random
import time

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from openai import OpenAI
from langchain.prompts import PromptTemplate


os.environ["OPENAI_API_BASE"] = 'URL'
os.environ["OPENAI_API_KEY"] = 'sk-xxxxx'
# print(os.environ)

# proxy = 'http://127.0.0.1:15777'
# os.environ['HTTP_PROXY'] = proxy
# os.environ['HTTPS_PROXY'] = proxy


client = OpenAI(
    base_url="URL", 
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(
        base_url="URL",
        follow_redirects=True,
    ),
)

prompt = PromptTemplate.from_template(
    '''
    Function Name: {target}
    Code Snippets:
        - "{snip_0}"
        - "{snip_1}"
        - "{snip_2}"
    Analysis Task: Based on the provided code snippets and the name of the function, conclude and explain the usage of this function into one sentence.
    '''
)


# Github Private Tokens
git_tokens = [
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
]



# Code Sequence Labeling Model
csl_tokenizer = AutoTokenizer.from_pretrained('/home/lhl/projects/models/hf/codebert-base')
csl_model = AutoModelForTokenClassification.from_pretrained(
    '/home/lhl/projects/checkpoints/csl_model_3_acc(0.967753354240161)', num_labels=3).cuda()


def concat(pred, tokens, cls: int):
    res = []
    i = 0
    flag = False
    temp = []
    while True:
        if flag is False and pred[i] == cls:
            temp = []
            temp.append(tokens[i])
            flag = True
        elif flag and pred[i] == cls:
            temp.append(tokens[i])
        elif flag and pred[i] != cls:
            flag = False
            res.append("".join(temp))
            temp = []
        i += 1
        if i >= 320:
            break

    return res

def code_sequence_label(code_tokens: list[str]):
    code_seq = ''.join(code_tokens)
    tokens = csl_tokenizer.tokenize(
        code_seq, 
        padding="max_length", 
        max_length=320, 
        truncation=True
    )
    
    inputs = csl_tokenizer(
        code_seq,
        padding="max_length",
        max_length=320,
        truncation=True,
        return_tensors="pt",
    )
    
    csl_model.eval()
    with torch.inference_mode():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        output = csl_model(**inputs).logits
        predict = output.argmax(dim=-1).squeeze(dim=0)
    
    """
    0. NONE
    1. API CALL
    2. IMPORTANT STRING
    """
    
    def filter(s: str):
        keys = [';', '(', ')', '{', '}', ',', '"', '\\', '/', '?', '!', '.', ':', '\'', '[', ']', '|', '-', '=', '+', '*', '%', '&']
        for k in keys:
            if k in s:
                s = s.replace(k, '')
        return s
    
    api_calls = [filter(s)
                 for s in concat(predict, tokens, 1) if len(s) >= 4]
    
    map = {}
    ac = []
    for s in api_calls:
        if s not in map and len(s) >= 5:
            map[s] = True
            ac.append(s)
            
    
    impo_strs = concat(predict, tokens, 2)
    
    return ac, impo_strs


def search_github_code(target, token, upper_limit=7, lower_limit=7,result_num = 3):

    # 设置请求标头
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # 搜索条件
    query = f"language:c+{target}"

    def find_and_print_context(multi_line_str, target_str):
        context_list = []
        # 找到目标字符串第一次出现的位置
        start_index = multi_line_str.find(target_str)

        # 如果找到了目标字符串
        if start_index != -1:
            # 找到目标字符串所在的行号
            line_number = multi_line_str.count('\n', 0, start_index) + 1

            # 将多行字符串分割成行列表
            lines = multi_line_str.split('\n')

            # 打印目标字符串的前 upper_limit 行和后 lower_limit 行
            start_print_index = max(0, line_number - upper_limit)
            end_print_index = min(len(lines), line_number + lower_limit)

            for i in range(start_print_index, end_print_index):
                context_list.append(lines[i] + '\n')

        return ''.join(context_list)

    # 发送请求
    response = requests.get(f"https://api.github.com/search/code?q={query}&per_page={str(result_num)}", headers=headers)

    # 检查请求是否成功
    if response.status_code == 200:
        results = response.json()
        # print(f"Found {results['total_count']} results")
        context_results = []
        for item in results['items']:
            # 获取代码文件内容
            file_response = requests.get(item['url'], headers=headers)
            if file_response.status_code == 200:
                file_content = file_response.json()['content']

                # 解码 Base64 编码的内容
                decoded_content = base64.b64decode(file_content).decode('utf-8')

                context_list = find_and_print_context(decoded_content, target)
                if context_list:
                    context_results.append(context_list)

        return 200, context_results
    else:
        # print(f"Failed to retrieve results. Status code: {response.status_code}")
        return 403, []
    
def online_retrieve(apis: list[str]):
    res = {}
    for api in apis:
        
        while True:
            token = git_tokens[random.randint(0, 9)]
            try: 
                code, context_info = search_github_code(api, token=token, upper_limit=10, lower_limit=5,result_num=3)
            
            except Exception as e:
                time.sleep(1.5)
                continue
            
            if code == 200:  
                res[api] = context_info
                break
            else:
                time.sleep(1.5)
    
    return res

def static_annotation_generate(src: dict) -> list[str]:
    res = []
    for api, ctx in src.items():
        try: 
                s0, s1, s2 = '', '', ''
                if len(ctx) >= 1:
                    s0 = ctx[0]
                if len(ctx) >= 2:
                    s1 = ctx[1]
                if len(ctx) >= 3:
                    s2 = ctx[2]
                
                query = prompt.format(target=api, snip_0=s0, snip_1=s1, snip_2=s2)
                
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": query}
                    ]
                )
                
                resp = completion.choices[0].message.content
                
                res.append(resp)
                
        except Exception as e:
            print(e)

    return res
