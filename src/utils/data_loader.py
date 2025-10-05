import json
from pathlib import Path
from typing import Dict, List, Any, Union


def load_json(file_path: Union[str, Path]) -> Union[Dict, List]:
    """
    JSON 파일을 로드합니다.
    
    Args:
        file_path: JSON 파일 경로
        
    Returns:
        Dict 또는 List - JSON 데이터
        
    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        json.JSONDecodeError: JSON 파싱 오류 시
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def save_json(data: Union[Dict, List], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    데이터를 JSON 파일로 저장합니다.
    
    Args:
        data: 저장할 데이터 (Dict 또는 List)
        file_path: 저장할 파일 경로
        indent: JSON 들여쓰기 (기본 2)
        
    Raises:
        IOError: 파일 저장 실패 시
    """
    file_path = Path(file_path)
    
    # 디렉토리가 없으면 생성
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    print(f"Saved to {file_path}")


def preprocess_for_importance(data: List[Dict[str, Any]]) -> List[str]:
    """
    중요도 측정을 위한 데이터 전처리
    
    Args:
        data: 원본 데이터 리스트
            [{"instruction": "...", "output": "..."}, ...]
        
    Returns:
        List[str] - 전처리된 텍스트 리스트
    """
    processed_texts = []
    
    for item in data:
        # instruction과 output 결합
        if 'instruction' in item and 'output' in item:
            text = f"{item['instruction']}\n{item['output']}"
        elif 'text' in item:
            text = item['text']
        else:
            # 다른 형식의 데이터 처리
            text = str(item)
        
        processed_texts.append(text)
    
    return processed_texts


if __name__ == "__main__":
    # 테스트 코드
    print("Testing data_loader functions...")
    print("=" * 50)
    
    # 1. save_json 테스트
    print("\n[Test 1] save_json")
    test_data = {
        "test_key": "test_value",
        "test_list": [1, 2, 3]
    }
    test_path = "results/test_save.json"
    save_json(test_data, test_path)
    
    # 2. load_json 테스트
    print("\n[Test 2] load_json")
    loaded_data = load_json(test_path)
    print(f"Loaded data: {loaded_data}")
    assert loaded_data == test_data, "Data mismatch!"
    print("Load test passed")
    
    # 3. preprocess_for_importance 테스트
    print("\n[Test 3] preprocess_for_importance")
    sample_data = [
        {"instruction": "What is AI?", "output": "AI is artificial intelligence."},
        {"instruction": "Explain ML", "output": "ML is machine learning."}
    ]
    processed = preprocess_for_importance(sample_data)
    print(f"Processed {len(processed)} items")
    print(f"Sample: {processed[0][:50]}...")
    
    print("\n" + "=" * 50)
    print("All tests passed!")