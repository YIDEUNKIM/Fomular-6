import csv
import time
from google import genai  
import re
from tqdm import tqdm

# 4개 API 키 (그대로 유지, 더 빠른 처리를 위해)
API_KEYS = [
    "AIzaSyBq7TQInFZe7xpXnMcwOex4Gvq0ncz91hg",
    "AIzaSyBlqfW2Cy5tehvGEsBS9_epLNy1SGwwdDQ",   
    "AIzaSyAp79T6wmsLf0Hw_RecuQkK_g0izA66pso",    
    "AIzaSyAhR5hBfutg8fQ9jh0uf0QK_H68gtceTk4",
]

clients = [genai.Client(api_key=key) for key in API_KEYS]

def translate_mega_batch_mednli(texts, dialect="Jeju", client_index=0):
    """MedNLI용 20개 텍스트를 한 번의 API 호출로 번역"""
    if not texts or all(not t for t in texts):
        return [""] * len(texts)
    
    # 유효한 텍스트만 필터링
    valid_texts = [t for t in texts if t and str(t).strip()]
    if not valid_texts:
        return [""] * len(texts)
    
    # 대규모 배치 프롬프트
    numbered_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(valid_texts)])
    
    prompt = f"""
    다음 {len(valid_texts)}개 문장을 {dialect} 방언으로 번역해주세요.
    각 문장을 개행으로 구분하고 번호를 반드시 유지해주세요.
    다른 설명은 절대 추가하지 말고 번역 결과만 출력해주세요.
    
    {numbered_texts}
    """
    
    try:
        client = clients[client_index % len(clients)]
        response = client.models.generate_content(
            model="gemini-2.5-pro", 
            contents=prompt
        )
        
        # 응답 파싱 (번호. 번역문 형식)
        return parse_mega_response_mednli(response.text, texts, valid_texts)
        
    except Exception as e:
        print(f"MedNLI 메가 배치 번역 실패: {e}")
        return texts  # 실패시 원본 반환

def parse_mega_response_mednli(response_text, original_texts, valid_texts):
    """MedNLI 대규모 배치 응답 파싱"""
    results = [""] * len(original_texts)
    
    # 번호 패턴으로 분리 (1. ..., 2. ..., 등)
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # "1. 번역문", "2. 번역문" 형식 파싱
        match = re.match(r'^(\d+)\.\s*(.+)$', line)
        if match:
            idx = int(match.group(1)) - 1  # 0-based index
            translated = match.group(2).strip()
            
            if idx < len(valid_texts):
                # 원본 텍스트 위치 찾기
                for i, original in enumerate(original_texts):
                    if original == valid_texts[idx] and not results[i]:
                        results[i] = translated
                        break
    
    # 번역 실패한 항목은 원본 유지
    for i in range(len(results)):
        if not results[i] and original_texts[i]:
            results[i] = original_texts[i]
            
    return results

def process_MedNLI_jeju_only(input_csv, output_csv, mega_batch_size=20):
    """MedNLI 제주 방언만 번역"""
    
    with open(input_csv, "r", encoding="utf-8") as infile, \
         open(output_csv, "w", encoding="utf-8", newline="") as outfile:
        
        reader = csv.DictReader(infile)
        data_rows = list(reader)
        
        # MedNLI 필드명 (ai_answer, result 추가)
        fieldnames = [
            "gold_label",
            "sentence1_jeju",  # 제주 방언만
            "sentence2_jeju",  # 제주 방언만
            "ai_answer",
            "result"
        ]
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        total_mega_batches = (len(data_rows) + mega_batch_size - 1) // mega_batch_size
        
        print(f"제주 방언 번역 시작: {len(data_rows)}행 데이터")
        
        for mega_batch_num in tqdm(range(total_mega_batches), 
                                 desc="[제주 방언] 번역 진행"):
            
            start_idx = mega_batch_num * mega_batch_size
            end_idx = min(start_idx + mega_batch_size, len(data_rows))
            mega_batch = data_rows[start_idx:end_idx]
            
            # 모든 텍스트 수집 (한 메가배치에 20행 × 2컬럼 = 40개 텍스트)
            all_texts = []
            text_positions = []  # (row_idx, text_type)
            
            for row_idx, row in enumerate(mega_batch):
                all_texts.append(row.get("sentence1_ko", ""))
                text_positions.append((row_idx, "sentence1"))
                
                all_texts.append(row.get("sentence2_ko", ""))
                text_positions.append((row_idx, "sentence2"))
            
            # 메가 배치 번역 (40개 텍스트를 한 번의 API 호출로)
            translated_texts = translate_mega_batch_mednli(all_texts, "Jeju", mega_batch_num)
            
            # 결과 저장
            batch_results = [{} for _ in range(len(mega_batch))]
            
            for (row_idx, text_type), translated in zip(text_positions, translated_texts):
                if text_type == "sentence1":
                    batch_results[row_idx]["sentence1_jeju"] = translated
                elif text_type == "sentence2":
                    batch_results[row_idx]["sentence2_jeju"] = translated
            
            # gold_label, ai_answer, result 컬럼 채우기
            for i, row in enumerate(mega_batch):
                batch_results[i]["gold_label"] = row.get("gold_label", "")
                batch_results[i]["ai_answer"] = row.get("ai_answer", "")
                batch_results[i]["result"] = row.get("result", "")
                writer.writerow(batch_results[i])
            
            # 진행 상황 출력
            if (mega_batch_num + 1) % 5 == 0:
                print(f"[제주 방언] {mega_batch_num + 1}/{total_mega_batches} 배치 완료")
            
            # 배치 간 대기 (API 제한 고려)
            if mega_batch_num < total_mega_batches - 1:
                time.sleep(2)  # 2초 대기
                
    print(f"[제주 방언] 번역 완료! 저장 위치: {output_csv}")

if __name__ == "__main__":
    print("=" * 60)
    print("MedNLI 제주 방언 번역 시작")
    print("4개 API + 대규모 배치(20) = 최적화 처리")
    print("예상 시간: 30-40분")
    print("=" * 60)
    
    # 제주 방언만 실행
    process_MedNLI_jeju_only("mednli_kor.csv", "mednli_jeju.csv")
    
    print("=" * 60)
    print("제주 방언 번역 완료!")
    print("생성된 파일: mednli_jeju.csv")
    print("=" * 60)