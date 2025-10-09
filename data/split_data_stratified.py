import json
import random
from pathlib import Path
from collections import defaultdict

def load_all_data():
    """
    6개 카테고리 파일을 모두 읽어서 합치기
    각 데이터에 카테고리 정보 추가
    """
    data_dir = Path('data/raw')
    all_data = []
    category_counts = defaultdict(int)
    
    # data/raw/ 폴더의 모든 JSON 파일 읽기
    json_files = sorted(data_dir.glob('*.json'))
    
    print(f"발견된 파일: {len(json_files)}개")
    
    for json_file in json_files:
        # 파일명에서 카테고리 추출 (예: "tomato_disease.json" -> "disease")
        category = json_file.stem  # 확장자 제외한 파일명
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 각 데이터에 카테고리 정보 추가
        for item in data:
            item['category'] = category
            all_data.append(item)
            category_counts[category] += 1
    
    print(f"\n카테고리별 데이터 개수:")
    total = 0
    for cat, count in sorted(category_counts.items()):
        print(f"  - {cat}: {count}개 ({count/sum(category_counts.values())*100:.1f}%)")
        total += count
    
    print(f"\n총 데이터: {total}개")
    
    return all_data, category_counts


def stratified_split(data, ratios, seed=42):
    """
    카테고리별로 균등하게 분할
    
    Args:
        data: 전체 데이터 리스트 (각 item에 'category' 필드 있어야 함)
        ratios: {'name': ratio} 형태의 딕셔너리
        seed: 랜덤 시드
    
    Returns:
        각 분할의 데이터를 담은 딕셔너리
    """
    random.seed(seed)
    
    # 카테고리별로 데이터 그룹화
    category_data = defaultdict(list)
    for item in data:
        category_data[item['category']].append(item)
    
    # 각 카테고리 내에서 섞기
    for category in category_data:
        random.shuffle(category_data[category])
    
    # 분할 결과 저장
    splits = {name: [] for name in ratios.keys()}
    
    # 각 카테고리에서 비율대로 분할
    for category, items in category_data.items():
        total = len(items)
        start_idx = 0
        
        print(f"\n{category} 분할 중 (총 {total}개):")
        
        for i, (split_name, ratio) in enumerate(ratios.items()):
            # 마지막 split은 남은 모든 데이터
            if i == len(ratios) - 1:
                end_idx = total
            else:
                end_idx = start_idx + int(total * ratio)
            
            split_data = items[start_idx:end_idx]
            splits[split_name].extend(split_data)
            
            print(f"  - {split_name}: {len(split_data)}개 ({len(split_data)/total*100:.1f}%)")
            
            start_idx = end_idx
    
    return splits


def save_splits(splits, output_dir='data/split'):
    """분할된 데이터 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("최종 분할 결과:")
    print("="*60)
    
    for split_name, split_data in splits.items():
        # 파일명 생성
        filename = f"{split_name}.json"
        filepath = output_path / filename
        
        # 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        # 카테고리별 분포 확인
        category_dist = defaultdict(int)
        for item in split_data:
            category_dist[item['category']] += 1
        
        print(f"\n{filename}: {len(split_data)}개")
        for cat, count in sorted(category_dist.items()):
            print(f"    - {cat}: {count}개 ({count/len(split_data)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("모든 데이터 저장 완료!")


def main():
    """메인 실행 함수"""
    
    # 1. 모든 데이터 로드
    print("="*60)
    print("데이터 로딩 중...")
    print("="*60)
    
    all_data, category_counts = load_all_data()
    
    # 2. 분할 비율 정의 (프루닝 40% / 파인튜닝 40% / 검증 20%)
    split_ratios = {
        'pruning_activation': 0.40,     # 40% - Activation 기반 중요도 측정용
        'finetuning_lora': 0.40,        # 40% - LoRA 파인튜닝용
        'final_validation': 0.20        # 20% - 최종 성능 평가용
    }
    
    print("\n" + "="*60)
    print("계층적 분할 시작 (각 카테고리에서 균등하게)")
    print("="*60)
    
    # 3. 분할 실행
    splits = stratified_split(all_data, split_ratios)
    
    # 4. 저장
    save_splits(splits)
    
    # 5. 검증
    print("\n" + "="*60)
    print("분할 검증")
    print("="*60)
    
    total_count = sum(len(s) for s in splits.values())
    print(f"원본 데이터: {len(all_data)}개")
    print(f"분할 후 합계: {total_count}개")
    
    if total_count == len(all_data):
        print("데이터 손실 없음!")
    else:
        print(f"데이터 개수 불일치! 차이: {len(all_data) - total_count}개")
    
    # 카테고리별 비율 검증
    print("\n각 분할의 카테고리 비율이 원본과 유사한지 확인:")
    original_ratios = {cat: count/len(all_data) 
                       for cat, count in category_counts.items()}
    
    for split_name, split_data in splits.items():
        split_dist = defaultdict(int)
        for item in split_data:
            split_dist[item['category']] += 1
        
        print(f"\n{split_name}:")
        for cat in original_ratios:
            split_ratio = split_dist[cat] / len(split_data) if split_data else 0
            original_ratio = original_ratios[cat]
            diff = abs(split_ratio - original_ratio) * 100
            status = "OK" if diff < 2 else "WARNING"  # 2% 이내면 OK
            print(f"  [{status}] {cat}: {split_ratio*100:.1f}% (원본: {original_ratio*100:.1f}%, 차이: {diff:.1f}%)")


if __name__ == "__main__":
    main()