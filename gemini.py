import csv
import time
from google import genai  
import multiprocessing
import os 
from tqdm import tqdm


# ✅ Gemini API 설정
Client = genai.Client(api_key="AIzaSyBq7TQInFZe7xpXnMcwOex4Gvq0ncz91hg")


# ✅ 방언 번역 함수 정의
def translate_dialect(text, dialect="Jeju"):
    user_messages = {
        "Jeju": "다음 문장을 제주도 방언으로 자연스럽게 번역해줘, 만약 전문 언어라 해석이 어렵다면 영어로 남겨줘",
        "Gyeongsang": "다음 문장을 경상도 방언으로 자연스럽게 번역해줘, 만약 전문 언어라 해석이 어렵다면 영어로 남겨줘",
        "Jeolla": "다음 문장을 전라도 사투리로 자연스럽게 번역해줘, 만약 전문 언어라 해석이 어렵다면 영어로 남겨줘",
        "Chungcheong": "다음 문장을 충청도 사투리로 자연스럽게 번역해줘. 만약 전문 언어라 해석이 어렵다면 영어로 남겨줘"
    }
    
    system_message = {
        "Jeju": "너는 제주도 방언 전문가야. 이제 부터 문장이 주어지면 해당 지역 방언으로 정확하게 번역해야 해,다른 설명은 절대 추가하지마",
        "Gyeongsang": "너는 경상도 방언 전문가야. 이제 부터 문장이 주어지면 해당 지역 방언으로 정확하게 번역해야 해,다른 설명은 절대 추가하지마",
        "Jeolla": "너는 전라도 방언 전문가야. 이제 부터 문장이 주어지면 해당 지역 방언으로 정확하게 번역해야 해,다른 설명은 절대 추가하지마",
        "Chungcheong": "너는 충청도 방언 전문가야. 이제 부터 문장이 주어지면 해당 지역 방언으로 정확하게 번역해야 해,다른 설명은 절대 추가하지마"
    }
    
    if not text or str(text).strip() == "":
        return ""
    
    
    user_prompt = f"{user_messages[dialect]}\n{text}"
    full_prompt = f"{system_message[dialect]}\n\n{user_prompt}"
    
    try:
        response = Client.models.generate_content(
         model="gemini-2.5-pro",contents = full_prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"번역 에러 발생 ({dialect}): {e}")
        return text

# ✅ 파일 처리(TruthfulQA) 
def process_TruthfulQA(input_csv, output_csv, dialect):
    
        with open(input_csv, "r", encoding="utf-8") as infile, \
             open(output_csv, "w", encoding="utf-8", newline="") as outfile:
            
            reader = csv.DictReader(infile)
            data_rows = list(reader)
            
            
            fieldnames = [f"question_{dialect}", f"mc1_choices_{dialect}","mc1_label",f"mc2_choices_{dialect}","mc2_label","ai_answer_mc1","mc1_result","ai_answer_mc2","mc2_result"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, row in enumerate(tqdm(data_rows, desc=f"[{dialect}]번역 진행")):
                
                question_dialect = translate_dialect(row.get("question_ko", ""), dialect)
                mc1_choice_dialect = translate_dialect(row.get("mc1_choices_ko", ""), dialect)
                mc2_choice_dialect = translate_dialect(row.get("mc2_choices_ko", ""), dialect)
                
                writer.writerow({
                    f"question_{dialect}": question_dialect,
                    f"mc1_choices_{dialect}": mc1_choice_dialect,
                    f"mc1_label":"",
                    f"mc2_choices_{dialect}": mc2_choice_dialect,
                    f"mc2_label":"",
                    f"ai_answer_mc1":"",
                    f"mc1_result":"",
                    f"ai_answer_mc2":"",
                    f"mc2_result":""
                })
                print(f"[{dialect}] {i+1}번째 문장 번역 완료")
                
                time.sleep(1.2)
                
        print(f"\n[{dialect}] 모든 번역 완료! 저장 위치: {output_csv}")
        

# ✅ 파일 처리(MedNLI)
def process_mednli(input_csv, output_csv, dialect):
    
        with open(input_csv, "r", encoding="utf-8") as infile, \
             open(output_csv, "w", encoding="utf-8", newline="") as outfile:
            
            reader = csv.DictReader(infile)
            print(f"[{dialect}] CSV 컬럼:", reader.fieldnames)
            
            
            fieldnames = ["gold_label", f"sentence1_{dialect}", f"sentence2_{dialect}","ai_answer","result"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, row in enumerate(reader):
                sentence1_dialect = translate_dialect(row.get("sentence1_ko", ""), dialect)
                sentence2_dialect = translate_dialect(row.get("sentence2_ko", ""), dialect)
                
                
                writer.writerow({
                    "gold_label":"",
                    f"sentence1_{dialect}": sentence1_dialect,
                    f"sentence2_{dialect}": sentence2_dialect,
    		        f"ai_answer":"",
		            f"result":""
                })
                print(f"[{dialect}] {i+1}번째 문장 번역 완료")
                time.sleep(1.2)
                
        print(f"\n[{dialect}] 모든 번역 완료! 저장 위치: {output_csv}")
        

# ✅ 메인 실행부
if __name__ == "__main__":
    
    try:
        if os.name == 'nt' or multiprocessing.get_start_method(allow_none=True) != 'spawn':
             multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
       
        pass

    dialects = ["Jeju", "Gyeongsang", "Jeolla", "Chungcheong"]   
    truthfulqa_tasks = [
        ("TruthfulQA_result-gpt4o-gpt4o.csv", f"truthfulqa_{dialect}-gemini-2.5-pro.csv", dialect) 
        for dialect in dialects]
    
    
    with multiprocessing.Pool(processes=len(dialects)) as pool:
        pool.starmap(process_TruthfulQA, truthfulqa_tasks)
        
        mednli_tasks = [
    ("mednli_kor.csv", f"mednli_{dialect}.gemini-2.5-pro.csv", dialect) 
    for dialect in dialects]
        
    
    with multiprocessing.Pool(processes=len(dialects)) as pool:
        pool.starmap(process_mednli, mednli_tasks)
    