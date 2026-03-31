import json
import logging
import re
from typing import Dict, Any, List
import requests
from bs4 import BeautifulSoup
import ollama

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 사용할 모델명 제한
MODEL_NAME = 'qwen2.5-coder:7b'

def file_ops(action: str, file_path: str, content: str = "") -> str:
    """
    파일 읽기/쓰기/수정을 수행하는 Tool
    """
    try:
        if action == "read":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif action == "write":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Success: File '{file_path}' written successfully."
        elif action == "append":
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(content)
            return f"Success: Content appended to file '{file_path}' successfully."
        else:
            return f"Error: Unsupported action '{action}'. Use 'read', 'write', or 'append'."
    except Exception as e:
        return f"Error: {str(e)}"

def search_namuwiki(keyword: str) -> str:
    """
    나무위키에서 키워드를 검색하여 본문 내용을 가져오는 Tool
    """
    url = f"https://namu.wiki/w/{keyword}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 404:
            return f"Error: '{keyword}' 항목을 나무위키에서 찾을 수 없습니다."
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 나무위키의 본문 내용을 포함하는 div 태그를 찾음
        # 구조가 복잡하므로, 텍스트가 많은 블록들을 모아서 추출
        content_blocks = []
        for p in soup.find_all(['div', 'span', 'p']):
            text = p.get_text(strip=True)
            if len(text) > 50: # 의미 있는 텍스트만 추출
                content_blocks.append(text)
                if len(content_blocks) >= 3: # 컨텍스트가 꽉 차지 않도록 3문단까지만 추출
                    break
                    
        result_text = " ".join(content_blocks)
        target_len = min(len(result_text), 800) # 최대 800자로 엄격히 제한하여 환각 방지
        result_text = result_text[:target_len] + ("..." if len(result_text) > target_len else "")
        if not result_text:
            return "Error: 본문 내용을 추출할 수 없습니다."
            
        return result_text
    except Exception as e:
        return f"Error: 나무위키 검색 중 오류 발생 - {str(e)}"

# 툴 정의 목록 (Qwen 모델이 이해할 수 있는 형식)
agent_tools = [
    {
        "type": "function",
        "function": {
            "name": "file_ops",
            "description": "로컬 파일 시스템에서 텍스트 파일을 읽거나, 새로 쓰거나, 기존 내용에 추가합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "write", "append"],
                        "description": "수행할 작업: 'read'(읽기), 'write'(쓰기 - 기존 내용 덮어씀), 'append'(추가 - 기존 내용 끝에 덧붙임)"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "작업할 파일의 경로 (예: 'result.txt')"
                    },
                    "content": {
                        "type": "string",
                        "description": "파일에 쓰거나 추가할 내용 (action이 'read'일 때는 무시됨)"
                    }
                },
                "required": ["action", "file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_namuwiki",
            "description": "나무위키(namu.wiki)에서 주어진 대상 키워드를 검색하여 해당 항목의 설명 텍스트를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "나무위키에서 검색할 키워드 (예: '젤다')"
                    }
                },
                "required": ["keyword"]
            }
        }
    }
]

# 툴 함수 매핑 딕셔너리
tool_functions = {
    "file_ops": file_ops,
    "search_namuwiki": search_namuwiki
}

def run_agent_turn(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    현재 메시지 히스토리를 바탕으로 모델에 요청을 보내고,
    필요한 도구들을 차례대로 모두 호출한 뒤 최종 텍스트 답변까지 완료하면
    업데이트된 메시지 히스토리를 반환합니다.
    """
    
    # 마지막 사용자 프롬프트 찾기 (System Helper 프롬프트용)
    last_user_prompt = ""
    for m in reversed(messages):
        if m["role"] == "user":
            last_user_prompt = m["content"]
            break

    MAX_TURNS = 4
    turns = 0

    while turns < MAX_TURNS:
        turns += 1
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                tools=agent_tools
            )
        except Exception as e:
            logger.error("Ollama API 호출 중 예외 발생: %s", str(e))
            break

        message = response['message']
        messages.append(message)
        
        # qwen2.5-coder:7b may return tool calls as JSON in content instead of tool_calls field
        tool_calls = message.get('tool_calls') or []
        content = message.get('content', '').strip()
        
        if not tool_calls:
            # 정규식을 사용하여 content 내에서 JSON 형태의 툴 호출을 추출
            matches = re.findall(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', content, re.DOTALL)
            if matches:
                logger.info(f"정규식 추출 결과: {matches}")
            for match in matches:
                try:
                    parsed = json.loads(match)
                    tool_calls.append({'function': parsed})
                except Exception as e:
                    logger.error(f"JSON 파싱 실패: {e} - 대상 문자열: {match}")

        # 모델이 tool 호출을 원하지 않으면 종료
        if not tool_calls:
            print(f"\n[AI]: {content}\n")
            break
            
        # 히스토리에 저장된 메시지를 단일 툴 호출로 정제 (모델 혼동 방지)
        if len(tool_calls) > 1:
            logger.info("여러 툴이 동시에 호출되었습니다. 첫 번째 툴만 실행하도록 강제합니다.")
            tool_calls = [tool_calls[0]]
            if message.get('tool_calls'):
                message['tool_calls'] = tool_calls
            else:
                message['content'] = json.dumps(tool_calls[0]['function'], ensure_ascii=False)
                
        # 모델이 툴을 호출하기로 한 경우 (첫 번째 툴만 실행)
        tool_call = tool_calls[0]
        function_name = tool_call['function']['name']
        arguments = tool_call['function']['arguments']
        
        logger.info(f"\n>>> Tool 호출 시도: {function_name} with arguments: {arguments}")
        
        if function_name in tool_functions:
            # 툴 실행
            func = tool_functions[function_name]
            try:
                result = func(**arguments)
                logger.info(f">>> Tool 실행 완료")
            except Exception as e:
                result = f"Error executing {function_name}: {str(e)}"
                logger.error(result)
            
            # 실행 결과를 메시지 기록에 'tool' 역할로 추가
            result_str = f"도구 실행 결과:\n{str(result)}"
            
            if function_name == "search_namuwiki":
                # 검색 결과 후 파일 쓰기 유도
                result_str += f"\n\n[System Helper: 위 검색 결과를 바탕으로 사용자의 원래 요청 '{last_user_prompt}'을(를) 완수하세요. 반드시 `{{\"name\": \"도구이름\", \"arguments\": {{...}}}}` 형태의 JSON만 출력해야 합니다. 일반 텍스트나 다른 형태의 JSON은 허용되지 않습니다.]"
            else:
                # 파일 쓰기 후 종료 유도
                result_str += f"\n\n[System Helper: 요청된 작업이 완료되었습니다. 더 이상 도구를 호출하지 말고, 사용자에게 완료되었다고 최종 답변(일반 텍스트)을 제공하세요.]"
                
            messages.append({
                "role": "tool",
                "content": result_str,
                "name": function_name
            })
        else:
            error_msg = f"Error: Unknown tool function '{function_name}'. 사용 가능한 도구는 오직 'search_namuwiki'와 'file_ops'뿐입니다. 올바른 이름으로 다시 호출하거나 작업을 종료하세요."
            logger.error(error_msg)
            messages.append({
                "role": "tool",
                "content": error_msg,
                "name": function_name
            })
        
        # 툴 실행 결과를 포함하여 다시 모델에 전달하고 다음 액션(또는 답변)을 기다림
        
    if turns >= MAX_TURNS:
        print("\n[시스템]: AI가 도구 호출 과정에서 길을 잃어(환각 현상) 연속 호출을 강제 종료합니다. 다른 키워드나 명확한 문장으로 다시 질문해 주세요.\n")

    return messages

def main():
    print("==================================================")
    print(" AI 에이전트와 대화를 시작합니다. (종료하려면 'exit' 입력)")
    print("==================================================\n")
    
    messages = [
        {
            "role": "system", 
            "content": "당신은 도구(Tools)를 활용하여 사용자와 대화하는 유능한 단일 AI 어시스턴트입니다. 사용자의 질문에 친절하게 답변하거나, 필요한 경우 도구(검색, 파일 작업 등)를 호출하여 문제를 해결하세요. \n[특별 규칙]: 만약 사용자가 'OOO에 대해 검색해줘'라고 요청한다면, 이는 2단계 작업입니다. 첫 번째 턴에는 반드시 'search_namuwiki' 도구를 호출하여 검색하고, 그 결과를 받은 후 두 번째 턴에는 자동으로 'file_ops' 도구를 호출하여 검색 결과를 'OOO.txt' 등의 파일로 저장해야 합니다. 일반 텍스트 답변은 지양하고 도구 호출이 필요할 때는 반드시 다음과 같은 JSON 형식으로만 한 줄씩 출력하세요: {\"name\": \"tool_name\", \"arguments\": {\"arg1\": \"value\"}}"
        }
    ]
    
    while True:
        try:
            user_input = input("사용자: ")
            if user_input.strip().lower() == 'exit':
                print("대화를 종료합니다.")
                break
            
            if not user_input.strip():
                continue
                
            messages.append({"role": "user", "content": user_input})
            messages = run_agent_turn(messages)
            
        except KeyboardInterrupt:
            print("\n대화를 종료합니다.")
            break

if __name__ == "__main__":
    main()
