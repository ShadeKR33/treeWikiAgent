import re
import json

content = '''{"name": "search_namuwiki", "arguments": {"keyword": "젤다"}}
{"name": "file_ops", "arguments": {"action": "write", "content": "", "file_path": "zelda.txt"}}'''

matches = re.findall(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', content, re.DOTALL)
print("Matches:", matches)
for m in matches:
    print(json.loads(m))
